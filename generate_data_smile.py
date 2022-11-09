# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semantics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path
import argparse
import pickle
import tensorflow as tf 
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm
from dnnlib import tflib
from PIL import Image  
from styleGAN2_model.model_settings import MODEL_POOL
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from interface.utils.logger import setup_logger
import editor
from interface.utils.myinverter import StyleGAN2Inverter
import glob
import time

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Generate images with given model.')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        choices=list(MODEL_POOL),
                        help='Name of the model for generation. (required)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    # parser.add_argument('-i', '--data_dir', type=str, default='./results/stylegan2_ffhq',
    #                     help='If specified, will load latent codes from given ')
    # parser.add_argument('-b', '--boundary_path', type=str,required=True,
    #                     help='Path to the semantic boundary. (required)')
    parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                        help='If specified, will load latent codes from given '
                             'path instead of randomly sampling. (optional)')
    parser.add_argument('-n', '--num', type=int, default=5,
                        help='Number of images to generate. This field will be '
                             'ignored if `latent_codes_path` is specified. '
                             '(default: 1)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP','s'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('-S', '--generate_style', action='store_true',
                        help='If specified, will generate layer-wise style codes '
                             'in Style GAN. (default: do not generate styles)')
    parser.add_argument('-I', '--generate_image', action='store_false',
                        help='If specified, will skip generating images in '
                             'Style GAN. (default: generate images)')
    parser.add_argument('-p', '--truncation_psi', type=float,default='0.8')
    parser.add_argument('--code_type',choices=['w','s','s_mean_std','s_flat','images_1K','images','images_10','images_10K','images_100K'],default='w')
    parser.add_argument('--resize',type=int,default=None,help='save image size')
    parser.add_argument('--lamda_attr',type=float,default=1e-2,
                        help='the weight of attribute loss')
    parser.add_argument('--lamda_norm',type=float,default=1e-6,
                        help='the weight of L2-norm loss')
    # parser.add_argument('--learning_rate', type=float, default=0.01,
    #                     help='Learning rate for optimization. (default: 0.01)')
    # parser.add_argument('--num_iterations', type=int, default=100,
    #                     help='Number of optimization iterations. (default: 100)')

    # parser.add_argument('--loss_weight_feat', type=float, default=1e-4,
    #                     help='The perceptual loss scale for optimization. '
    #                          '(default: 5e-5)')
    return parser.parse_args()

def compute_loss(lamda,S,Z):
    loss=(1-lamda)*Z+lamda*S
    return loss

def LoadModel(model_name):
    tflib.init_tf()
    tmp=MODEL_POOL[model_name]['tf_model_path']
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    return Gs

def GetImg(Gs,num_img,num_once,output_path,resize=None):
    print('Generate Image')
    tmp='./results/stylegan2_ffhq'+'/W.npy'
    dlatents=np.load(tmp) 
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    all_images=[]
    for i in range(int(num_img/num_once)):
        print(i)
        images=[]
        for k in range(num_once):
            tmp=dlatents[i*num_once+k]
            tmp=tmp[None,None,:]
            tmp=np.tile(tmp,(1,Gs.components.synthesis.input_shape[1],1))
            image2= Gs.components.synthesis.run(tmp, randomize_noise=False, output_transform=fmt)
            
            if resize is not None:
                img=Image.fromarray(image2[0]).resize((resize,resize),Image.LANCZOS)
                img=np.array(img)
                image2=img[None,:]
            
            images.append(image2)
            
        images=np.concatenate(images)
        
        all_images.append(images)
        
    all_images=np.concatenate(all_images)
    
    return all_images

def SelectName(layer_name,suffix):
    if suffix==None:
        tmp1='add:0' in layer_name 
        tmp2='shape=(?,' in layer_name
        tmp4='G_synthesis_1' in layer_name
        tmp= tmp1 and tmp2 and tmp4  
    else:
        tmp1=('/Conv0_up'+suffix) in layer_name 
        tmp2=('/Conv1'+suffix) in layer_name 
        tmp3=('4x4/Conv'+suffix) in layer_name 
        tmp4='G_synthesis_1' in layer_name
        tmp5=('/ToRGB'+suffix) in layer_name
        tmp= (tmp1 or tmp2 or tmp3 or tmp5) and tmp4 
    return tmp

def GetSNames(suffix):
    #get style tensor name 
    with tf.Session() as sess:
        op = sess.graph.get_operations()
    layers=[m.values() for m in op]
    
    
    select_layers=[]
    for layer in layers:
        layer_name=str(layer)
        if SelectName(layer_name,suffix):
            select_layers.append(layer[0])
    return select_layers

def SelectName2(layer_name):
    tmp1='mod_bias' in layer_name 
    tmp2='mod_weight' in layer_name
    tmp3='ToRGB' in layer_name 
    
    tmp= (tmp1 or tmp2) and (not tmp3) 
    return tmp

def GetS(output_path,num_img):
    print('Generate S')
    tmp='results/stylegan2_ffhq'+'/W.npy'
    dlatents=np.load(tmp)[:num_img]
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        Gs=LoadModel(model_name)
        Gs.print_layers()  #for ada
        select_layers1=GetSNames(suffix=None)  #None,'/mul_1:0','/mod_weight/read:0','/MatMul:0'
        dlatents=dlatents[:,None,:]
        dlatents=np.tile(dlatents,(1,Gs.components.synthesis.input_shape[1],1))
        
        all_s = sess.run(
            select_layers1,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatents})
    
    layer_names=[layer.name for layer in select_layers1]
    save_tmp=[layer_names,all_s]
    return save_tmp

def GetS_Z(output_path,num_img):
    print('Generate S')
    tmp_s='results/stylegan2_ffhq'+'/S'
    tmp='results/stylegan2_ffhq'+'/W.npy'
    dlatents=np.load(tmp)[:num_img]
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        Gs=LoadModel(model_name)
        Gs.print_layers()  #for ada
        select_layers1=GetSNames(suffix=None)  #None,'/mul_1:0','/mod_weight/read:0','/MatMul:0'
        dlatents=dlatents[:,None,:]
        dlatents=np.tile(dlatents,(1,Gs.components.synthesis.input_shape[1],1))
        
        all_s = sess.run(
            select_layers1,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatents})
    
    layer_names=[layer.name for layer in select_layers1]
    save_tmp=[layer_names,all_s]
    return save_tmp

def GetCodeMS(dlatents):
        m=[]
        std=[]
        for i in range(len(dlatents)):
            tmp= dlatents[i] 
            tmp_mean=tmp.mean(axis=0)
            tmp_std=tmp.std(axis=0)
            m.append(tmp_mean)
            std.append(tmp_std)
        return m,std



def main():
    """Main function."""
    args = parse_args()
    if args.output_dir is None:
        output_dir='./results/'+model_name
    else:
        output_dir=args.output_dir
    if not os.path.isdir(output_dir):
        os.system('mkdir '+output_dir)
    print('output_dir:',output_dir)
    logger = setup_logger(output_dir, logger_name='generate_data')

    logger.info(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger, truncation_psi=args.truncation_psi)
    kwargs = {'latent_space_type': args.latent_space_type}

    logger.info(f'Preparing latent codes.')
    if args.code_type=='w':
        # Gs=LoadModel(dataset_name=dataset_name)
        # dlatents=GetCode(Gs,random_state,num_img,num_once,truncation)
        
        # tmp=output_path+'/W'
        # np.save(tmp,dlatents)

        if os.path.isfile(args.latent_codes_path):
            logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
            latent_codes = np.load(args.latent_codes_path)
            latent_codes = model.preprocess(latent_codes, **kwargs)
        else:
            logger.info(f'  Sample latent codes randomly.')
            latent_codes = model.easy_sample(args.num, **kwargs)
        total_num = latent_codes.shape[0]

        logger.info(f'Generating {total_num} samples.')
        results = defaultdict(list)
        pbar = tqdm(total=total_num, leave=False)

        for latent_codes_batch in model.get_batch_inputs(latent_codes):
            # outputs = model.easy_synthesize(latent_codes_batch,
            #                                 **kwargs,
            #                                 generate_style=args.generate_style,
            #                                 generate_image=args.generate_image)
            outputs = model.easy_style_mixing(latent_codes_batch,
                                            style_range=range(8,18),
                                            style_codes=None,
                                            mix_ratio=0.6,
                                            **kwargs,
                                            generate_style=args.generate_style,
                                            generate_image=args.generate_image)
            for key, val in outputs.items():
                if key == 'image':
                    for image in val:
                        save_path = os.path.join(output_dir, f'{pbar.n:06d}.jpg')
                        cv2.imwrite(save_path, image[:, :, ::-1])
                        pbar.update(1)
                else:
                    results[key].append(val)
            if 'image' not in outputs:
                pbar.update(latent_codes_batch.shape[0])
            if pbar.n % 1000 == 0 or pbar.n == total_num:
                logger.debug(f'  Finish {pbar.n:6d} samples.')
        pbar.close()

        logger.info(f'Saving results.')
        for key, val in results.items():
            if key=='s':
                for s_i in range(26):
                    save_path = os.path.join(output_dir, f'{key}_{s_i}.npy')
                    s_latent=np.concatenate([v[s_i] for v in val], axis=0)
                    print(s_latent.shape)
                    np.save(save_path,s_latent )
                    s_mean=s_latent.mean(axis=0)
                    s_std=s_latent.std(axis=0)
                    mean_save_path = os.path.join(output_dir, f'{key}_{s_i}_mean.npy')
                    std_save_path = os.path.join(output_dir, f'{key}_{s_i}_std.npy')
                    np.save(mean_save_path, s_mean)
                    np.save(std_save_path, s_std)
            else:
                save_path = os.path.join(output_dir, f'{key}.npy')
                np.save(save_path, np.concatenate(val, axis=0))
                print(np.concatenate(val, axis=0).shape)
    
    elif args.code_type=='s':
        save_name='S'
        save_tmp=GetS_Z(output_dir,num_img=args.num)
        tmp=output_dir+'/'+save_name
        with open(tmp, "wb") as fp:
            pickle.dump(save_tmp, fp)
    
        
    elif args.code_type=='s_mean_std':
        save_tmp=GetS_Z(output_dir,num_img=args.num)
        
        dlatents=save_tmp[1]
        m,std=GetCodeMS(dlatents)
        save_tmp=[m,std]
        save_name='S_mean_std'
        tmp=output_dir+'/'+save_name
        with open(tmp, "wb") as fp:
            pickle.dump(save_tmp, fp)
            
    elif args.code_type=='s_flat':
        save_tmp=GetS(output_dir,num_img=args.num)
        dlatents=save_tmp[1]
        dlatents=np.concatenate(dlatents,axis=1)
        
        tmp=output_dir+'/S_Flat'
        np.save(tmp,dlatents)

    elif args.code_type=='images':
        Gs=LoadModel(model_name=model_name)
        save_name='images'
        all_images=GetImg(Gs,num_img=args.num,num_once=2,output_path=output_dir,resize=args.resize)
        tmp=output_dir+'/'+save_name
        with open(tmp, "wb") as fp:
            pickle.dump(all_images, fp)

if __name__ == '__main__':
    model_name = 'stylegan2_ffhq'
    # dataset_name=MODEL_POOL[model_name]['tf_model_path']
    main()
