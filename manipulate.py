import os
import os.path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import torch
import cv2
import pickle
import numpy as np
import tensorflow as tf
from dnnlib import tflib
from interface.utils.visualizer import HtmlPageVisualizer
from styleGAN2_model.model_settings import MODEL_POOL,MODEL_DIR
from PIL import Image
import matplotlib.pyplot as plt
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from s_advance import StyleGAN2Advance
import glob
import time
from skimage.color import gray2rgb,rgb2gray
# torch.backends.cudnn.enabled = False


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('-i', '--data_dir', type=str, required=True,
                        help='If specified, will load latent codes from given ')

    parser.add_argument('-b', '--boundary_path', type=str,
                        required=True,
                        help='Path to the semantic boundary. (required)')

    parser.add_argument('--alpha', type=float, default=3.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')

    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('--code_type',choices=['w','s','sp','s_mean_std','s_flat','images_1K','images','images_10','images_10K','images_100K'],default='w')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization. (default: 0.01)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations. (default: 100)')

    parser.add_argument('--loss_weight_feat', type=float, default=1e-4,
                        help='The perceptual loss scale for optimization. '
                             '(default: 5e-5)')
    parser.add_argument('-S', '--generate_style', action='store_true',
                        help='If specified, will generate layer-wise style codes '
                             'in Style GAN. (default: do not generate styles)')
    parser.add_argument('-I', '--generate_image', action='store_false',
                        help='If specified, will skip generating images in '
                             'Style GAN. (default: generate images)')
    return parser.parse_args()

def Vis(bname,suffix,out,rownames=None,colnames=None):
    num_images=out.shape[0]
    step=out.shape[1]
    
    if colnames is None:
        colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    if rownames is None:
        rownames=[str(i) for i in range(num_images)]
    
    
    visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 1, viz_size=256)
    visualizer.set_headers(
      ['Name'] +colnames)
    
    for i in range(num_images):
        visualizer.set_cell(i, 0, text=rownames[i])
    
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            visualizer.set_cell(i, 1+k, image=image)
    
    # Save results.
    visualizer.save(f'./html/'+bname+'_'+suffix+'.html')




def LoadData(img_path):
    tmp=img_path+'S'
    with open(tmp, "rb") as fp:   #Pickling
        s_names,all_s=pickle.load( fp)
    dlatents=all_s
 
    pindexs=[]
    mindexs=[]
    for i in range(len(s_names)):
        name=s_names[i]
        if not('ToRGB' in name):
            mindexs.append(i)
        else:
            pindexs.append(i)
    
    tmp=img_path+'S_mean_std'
    with open(tmp, "rb") as fp:   #Pickling
        m,std=pickle.load( fp)
 
    return dlatents,s_names,mindexs,pindexs,m,std


def LoadModel(model_path,model_name):
    tflib.init_tf()
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    Gs.print_layers()
    return Gs


def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    if nchw_to_nhwc:
        images = np.transpose(images, [0, 2, 3, 1])
    
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    
    np.clip(images, 0, 255, out=images)
    images=images.astype('uint8')
    return images


def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        images=np.rollaxis(images, 3, 1)
    return images/ 255 *(drange[1] - drange[0])+ drange[0]

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run():
    
    latent_space_type = args.latent_space_type

    assert os.path.exists(args.data_dir), f'data_dir {args.data_dir} dose not exist!'
    origin_img_dir='./results/stylegan2_ffhq/origin'
    code_dir='./results/stylegan2_ffhq'
    diffuse_code_dir = os.path.join(args.data_dir, 'diffuse_code')
    res_dir = os.path.join(args.data_dir, 'diffuse_res')
    assert os.path.exists(origin_img_dir), f'{origin_img_dir} dose not exist!'
    assert os.path.exists(code_dir), f'data_dir {code_dir} dose not exist!'
    mkdir(res_dir)
    mkdir(diffuse_code_dir)

    print(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger=None)
    kwargs = {'latent_space_type': latent_space_type}

    print(f'Initializing Advance.')
    advance = StyleGAN2Advance(
        model_name,
        learning_rate=args.learning_rate,
        iteration=args.num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=args.loss_weight_feat,
        logger=None,
        stylegan2_model=model)

    print(f'Preparing boundary.')
    boundary_path=args.boundary_path
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{boundary_path}` does not exist!')
    boundary = np.load(boundary_path)
    # boundary=boundary[0]

    print(f'Load latent codes and images from `{args.data_dir}`.')
    latent_codes = []
    origin_img_list = []
    for img in glob.glob(os.path.join(origin_img_dir, '*'))[::-1]:
        code_path = './results/stylegan2_ffhq/w.npy'
        if os.path.exists(code_path):
            latent_codes.append(code_path)
            origin_img_list.append(img)
    total_num = len(latent_codes)

    print(f'Processing {total_num} samples.')
    times = []
    img_index=0
    latent_codes=np.load(code_path)
    latent_codes = latent_codes.reshape(-1, 512)
    for latent_codes_batch in model.get_batch_inputs(latent_codes):
        image_name = os.path.splitext(os.path.basename(origin_img_list[img_index]))[0]

        if os.path.exists(os.path.join(code_dir, f'{image_name}_advance_wp.npy')):
            continue

        origin_img=Image.open(origin_img_list[img_index]).convert("RGB").resize((256,256))
        origin_img=np.array(origin_img)

        neck_mask=np.load('./results/semantic_mask.npy')
        neck_mask=Image.fromarray(neck_mask[6]).resize((256,256))
        neck_mask=np.array(neck_mask)
        neck_mask=gray2rgb(neck_mask)
        neck_mask = (neck_mask > 0).astype(np.uint8) * 255

        mask_dilate = cv2.dilate(neck_mask, kernel=np.ones((30, 30), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(35, 35))
        mask_dilate_blur = neck_mask + (255 - neck_mask) // 255 * mask_dilate_blur
        
        train_count = 0
        edited_img=Image.fromarray(out[0,0]).resize((256,256))
        edited_img = edited_img.convert("RGB")
        edited_img=np.array(edited_img)      

        synthesis_image = origin_img * (1 - neck_mask // 255) + \
                          edited_img * (neck_mask // 255)
        target_image = synthesis_image[:, :, ::-1]
        target_image=rgb2gray(target_image).astype('uint8')
        target_image=target_image.reshape(256,256,1)
        
        start_diffuse = time.clock()
        code, viz_result ,lam= advance.easy_mask_diffuse(target=target_image,
                                                      init_code=latent_codes_batch,#(1,512)
                                                      mask=mask_dilate_blur,
                                                      lamda=boundary_tmp,
                                                      dlatents=boundary,
                                                      **kwargs)

        time_diffuse = (time.clock() - start_diffuse)

        times.append(time_diffuse)
        viz_result = viz_result[:, :, ::-1]
        res = origin_img * (1 - mask_dilate_blur // 255) + viz_result * (mask_dilate_blur // 255)
        print('train %d times.' % train_count)
        np.save(os.path.join(diffuse_code_dir, f'{image_name}_advance_sp.npy'), code)
        cv2.imwrite(os.path.join(res_dir, f'{image_name}.jpg'), res)
        img_index+=1
        break
    return lam


class Manipulator():
    def __init__(self,dataset_name='stylegan2_ffhq'):
        self.file_path='./'
        self.img_path=self.file_path+'results/'+dataset_name+'/'
        self.model_path=MODEL_DIR+'/'
        self.dataset_name=dataset_name
        self.model_name=dataset_name+'.pkl'
        
        self.alpha=[0] #manipulation strength 
        self.num_images=10
        self.img_index=0  #which image to start 
        self.viz_size=256
        self.manipulate_layers=None #which layer to manipulate, list
        
        self.dlatents,self.s_names,self.mindexs,self.pindexs,self.code_mean,self.code_std=LoadData(self.img_path)

        self.sess=tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.Gs=LoadModel(self.model_path,self.model_name)
        self.num_layers=len(self.dlatents)
        
        self.Vis=Vis
        self.noise_constant={}
        
        for i in range(len(self.s_names)):
            tmp1=self.s_names[i].split('/')
            if not 'ToRGB' in tmp1:
                tmp1[-1]='random_normal:0'
                size=int(tmp1[1].split('x')[0])
                tmp1='/'.join(tmp1)
                tmp=(1,1,size,size)
                self.noise_constant[tmp1]=np.random.random(tmp)
        
        tmp=self.Gs.components.synthesis.input_shape[1]
        d={}
        d['G_synthesis_1/dlatents_in:0']=np.zeros([1,tmp,512])
        names=list(self.noise_constant.keys())
        tmp=tflib.run(names,d)
        for i in range(len(names)):
            self.noise_constant[names[i]]=tmp[i]
        
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.img_size=self.Gs.output_shape[-1]
    
    def GenerateImg(self,codes):
        

        num_images,step=codes[0].shape[:2]

            
        out=np.zeros((num_images,step,self.img_size,self.img_size,3),dtype='uint8')
        for i in range(num_images):
            for k in range(step):
                d={}
                for m in range(len(self.s_names)):
                    d[self.s_names[m]]=codes[m][i,k][None,:]  #need to change
                d['G_synthesis_1/4x4/Const/Shape:0']=np.array([1,18,  512], dtype=np.int32)
                d.update(self.noise_constant)
                img=tflib.run('G_synthesis_1/images_out:0', d)
                image=convert_images_to_uint8(img, nchw_to_nhwc=True)
                out[i,k,:,:,:]=image[0]
        return out
    
    
    
    def MSCode(self,dlatent_tmp,boundary_tmp):
        
        step=len(self.alpha)
        dlatent_tmp1=[tmp.reshape((self.num_images,-1)) for tmp in dlatent_tmp]
        dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1] # (10, 7, 512)

        l=np.array(self.alpha)
        l=l.reshape(
                    [step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
        
        if type(self.manipulate_layers)==int:
            tmp=[self.manipulate_layers]
        elif type(self.manipulate_layers)==list:
            tmp=self.manipulate_layers
        elif self.manipulate_layers is None:
            tmp=np.arange(len(boundary_tmp))
        else:
            raise ValueError('manipulate_layers is wrong')
        for i in tmp:
            dlatent_tmp2[i]+=l*boundary_tmp[i]
        
        codes=[]
        for i in range(len(dlatent_tmp2)):
            tmp=list(dlatent_tmp[i].shape)
            tmp.insert(1,step)
            codes.append(dlatent_tmp2[i].reshape(tmp))
        return codes
    
    
    def EditOne(self,bname,dlatent_tmp=None):
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[]
        for i in range(len(self.boundary)):
            tmp=self.boundary[i]
            if len(tmp)<=bname:
                boundary_tmp.append([])
            else:
                boundary_tmp.append(tmp[bname])
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
            
        out=self.GenerateImg(codes)
        return codes,out
    
    def EditOneC(self,cindex,dlatent_tmp=None): 
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[[] for i in range(len(self.dlatents))]
        
        #'only manipulate 1 layer and one channel'
        assert len(self.manipulate_layers)==1 
        
        ml=self.manipulate_layers[0]
        tmp=dlatent_tmp[ml].shape[1] #ada
        tmp1=np.zeros(tmp)
        tmp1[cindex]=self.code_std[ml][cindex]  #1
        boundary_tmp[ml]=tmp1
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
        out=self.GenerateImg(codes)
        return codes,out,boundary_tmp
    
        
    def W2S(self,dlatent_tmp):
        
        all_s = self.sess.run(
            self.s_names,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatent_tmp})
        return all_s
        
    
    
    
    
    


#%%
if __name__ == "__main__":
    model_name = 'stylegan2_ffhq'
    args = parse_args()
    M=Manipulator(dataset_name='stylegan2_ffhq')
    alpha = 3 #@param {type:"slider", min:-10, max:10, step:0.1}
    M.img_index=0   #index for different images
    M.num_images=1  
    lindex,cindex=6,501 #(layer index, channel index), please copy from configs in above
    if args.code_type=='sp':
        M.alpha=[alpha]
        M.manipulate_layers=[lindex]
        codes,out,boundary_tmp=M.EditOneC(cindex) 
        dlatents_=M.dlatents
        lam=run()

        M.alpha=[0]
        codes,out,lam=M.EditOneC(cindex,lam)
        original=Image.fromarray(out[0,0]).resize((256,256))
        original = original.convert("RGBA")


        M.alpha=[-alpha,alpha]
        M.manipulate_layers=[lindex]
        codes,out,_=M.EditOneC(cindex) 
        positive=Image.fromarray(out[0,1]).resize((256,256))
        negative=Image.fromarray(out[0,0]).resize((256,256))

    elif args.code_type=='s':
        M.alpha=[0]
        M.manipulate_layers=[lindex]
        codes,out,boundary_tmp=M.EditOneC(cindex) 
        original=Image.fromarray(out[0,0]).resize((256,256))
        original = original.convert("RGBA")


        M.alpha=[-alpha,alpha]
        M.manipulate_layers=[lindex]
        codes,out,_=M.EditOneC(cindex) 
        positive=Image.fromarray(out[0,1]).resize((256,256))
        negative=Image.fromarray(out[0,0]).resize((256,256))

    plt.figure(figsize=(20,5), dpi= 100)
    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title('original')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(positive)
    plt.title('positive manipulation')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(negative)
    plt.title('negative manipulation')
    plt.axis('off')
    plt.savefig('./results/smile.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




