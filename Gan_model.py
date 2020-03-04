from __future__ import print_function
import os
import glob
import scipy

import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt


class Arguments(object):
    data_path = 'celebA'                     #path to CelebA dataset
    save_path = ''                           #path to save preprocessed image folder
    preproc_foldername = 'preprocessed'      #folder name for preprocessed images
    image_size = 64                          #images are resized to image_size value 
    num_images = 202590                      #the number of training images
    batch_size = 64                          #batch size
    dim_z = 100                              #the dimension of z variable (the generator input dimension)        
    n_g_filters = 64                         #the number of the generator filters (gets multiplied between layers)
    n_f_filters = 64                         #the number of the discriminator filters (gets multiplied between layers)           
    n_epoch = 25                             #the number of epochs
    lr = 0.0002                              #learning rate
    beta1 = 0.5                              #beta_1 parameter of Adam optimizer
    beta2 = 0.99                             #beta_2 parameter of Adam optimizer

args = Arguments()

#contains functions that load, preprocess and visualize images. 
#You will need to use preprocess_and_save_images(self, dir_name, save_path='') to preprocess data,
#then get_nextbatch(self, batch_size) generator to get batches 

class Dataset(object):     
    def __init__(self, data_path, num_imgs, target_imgsize):
        self.data_path = data_path
        self.num_imgs = num_imgs 
        self.target_imgsize = target_imgsize 
    
    def normalize_np_image(self, image):
        return (image / 255.0 - 0.5) / 0.5
    
    def denormalize_np_image(self, image):
        return (image * 0.5 + 0.5) * 255
    
    def get_input(self, image_path):
        image = np.array(Image.open(image_path)).astype(np.float32)
        return self.normalize_np_image(image)
    
    def get_imagelist(self, data_path, celebA=False): 
        if celebA == True:
            imgs_path = os.path.join(data_path, 'img_align_celeba/*.jpg')
        else:
            imgs_path = os.path.join(data_path, '*.jpg') 
        all_namelist = glob.glob(imgs_path, recursive=True)
        return all_namelist[:self.num_imgs]
    
    def load_and_preprocess_image(self, image_path): 
        image = Image.open(image_path)
        j = (image.size[0] - 100) // 2
        i = (image.size[1] - 100) // 2
        image = image.crop([j, i, j + 100, i + 100])    
        image = image.resize([self.target_imgsize, self.target_imgsize], Image.BILINEAR)
        image = np.array(image.convert('RGB')).astype(np.float32)
        image = self.normalize_np_image(image)
        return image    
    
    #reads data, preprocesses and saves to another folder with the given path. 
    def preprocess_and_save_images(self, dir_name, save_path=''): 
        preproc_folder_path = os.path.join(save_path, dir_name)
        if not os.path.exists(preproc_folder_path):
            os.makedirs(preproc_folder_path)   
            imgs_path = os.path.join(self.data_path, 'img_align_celeba/*.jpg')
            print('Saving and preprocessing images ...')
            for num, imgname in enumerate(glob.iglob(imgs_path, recursive=True)):
                cur_image = self.load_and_preprocess_image(imgname)
                cur_image = Image.fromarray(np.uint8(self.denormalize_np_image(cur_image)))
                cur_image.save(preproc_folder_path + '/preprocessed_image_%d.jpg' %(num)) 
        self.data_path= preproc_folder_path
            
    def get_nextbatch(self, batch_size): 
        assert (batch_size > 0),"Give a valid batch size"
        cur_idx = 0
        image_namelist = self.get_imagelist(self.data_path)
        while cur_idx + batch_size <= self.num_imgs:
            cur_namelist = image_namelist[cur_idx:cur_idx + batch_size]
            cur_batch = [self.get_input(image_path) for image_path in cur_namelist]
            cur_batch = np.array(cur_batch).astype(np.float32)
            cur_idx += batch_size
            yield cur_batch
      
    def show_image(self, image, normalized=True):
        if not type(image).__module__ == np.__name__:
            image = image.numpy()
        if normalized:
            npimg = (image * 0.5) + 0.5 
        npimg.astype(np.uint8)
        plt.imshow(npimg, interpolation='nearest')
        
def generator(x, args, reuse=False):
    with tf.device('/gpu:0'):
        with tf.variable_scope("generator", reuse=reuse): 
            #Layer Block 1
            with tf.variable_scope("layer1"):
                deconv1 = tf.layers.conv2d_transpose(inputs=x, 
                                             filters= args.n_g_filters*8, 
                                             kernel_size=4, 
                                             strides=1,
                                             padding='valid',
                                             use_bias=False,
                                             name='deconv')
                batch_norm1=tf.layers.batch_normalization(deconv1,
                                             name = 'batch_norm')
                relu1 = tf.nn.relu(batch_norm1, name='relu')
            #Layer Block 2
            with tf.variable_scope("layer2"):
                deconv2 = tf.layers.conv2d_transpose(inputs=relu1, 
                                             filters=args.n_g_filters*4, 
                                             kernel_size=4,
                                             strides=2,
                                             padding='same', 
                                             use_bias=False,
                                             name='deconv')
                batch_norm2 = tf.layers.batch_normalization(deconv2,
                                             name = 'batch_norm')
                relu2 = tf.nn.relu(batch_norm2, name='relu')
            #Layer Block 3
            with tf.variable_scope("layer3"):
                deconv3 = tf.layers.conv2d_transpose(inputs=relu2, 
                                             filters=args.n_g_filters*2, 
                                             kernel_size=4, 
                                             strides=2, 
                                             padding='same',
                                             use_bias = False,
                                             name='deconv')
                batch_norm3 = tf.layers.batch_normalization(deconv3, 
                                             name = 'batch_norm')
                relu3 = tf.nn.relu(batch_norm3, name='relu')
            #Layer Block 4
            with tf.variable_scope("layer4"):
                deconv4 = tf.layers.conv2d_transpose(inputs=relu3, 
                                             filters=args.n_g_filters, 
                                             kernel_size=4, 
                                             strides=2,
                                             padding='same',
                                             use_bias=False,
                                             name='deconv')
                batch_norm4 = tf.layers.batch_normalization(deconv4,
                                             name = 'batch_norm')
                relu4 = tf.nn.relu(batch_norm4, name='relu')
            #Output Layer
            with tf.variable_scope("last_layer"):
                logit = tf.layers.conv2d_transpose(inputs=relu4, 
                                             filters=3, 
                                             kernel_size=4, 
                                             strides=2, 
                                             padding='same',
                                             use_bias=False,
                                             name='logit')
                output = tf.nn.tanh(logit) 
    return output, logit


def discriminator(x, args, reuse=False):
    with tf.device('/gpu:0'):
        with tf.variable_scope("discriminator", reuse=reuse): 
            with tf.variable_scope("layer1"):
                conv1 = tf.layers.conv2d(inputs=x,
                                         filters=args.n_f_filters,
                                         kernel_size=4,
                                         strides=2,
                                         padding='same',
                                         use_bias=False,
                                         name='conv')
                relu1 = tf.nn.leaky_relu(conv1, alpha=0.2, name='relu')
            with tf.variable_scope("layer2"):
                conv2 = tf.layers.conv2d(inputs=relu1,
                                         filters=args.n_f_filters*2,
                                         kernel_size=4,
                                         strides=2,
                                         padding='same',
                                         use_bias=False,
                                         name='conv')
                batch_norm2 = tf.layers.batch_normalization(conv2,name='batch_norm')
                relu2 = tf.nn.leaky_relu(batch_norm2, alpha=0.2, name='relu')
            with tf.variable_scope("layer3"):
                conv3 = tf.layers.conv2d(inputs=relu2,
                                         filters=args.n_f_filters*4,
                                         kernel_size=4,
                                         strides=2,
                                         padding='same',
                                         use_bias=False,
                                         name='conv')
                batch_norm3 = tf.layers.batch_normalization(conv3, name='batch_norm')
                relu3 = tf.nn.leaky_relu(batch_norm3, name='relu')
            with tf.variable_scope("layer4"):
                conv4 = tf.layers.conv2d(inputs=relu3,
                                         filters=args.n_f_filters*8,
                                         kernel_size=4,
                                         strides=2,
                                         padding='same',
                                         use_bias=False,
                                         name='conv')
                batch_norm4 = tf.layers.batch_normalization(conv4, name='batch_norm')
                relu4 = tf.nn.leaky_relu(batch_norm4, alpha=0.2, name='relu')
            with tf.variable_scope("last_layer"):
                logit = tf.layers.conv2d(inputs=relu4,
                                         filters=1,
                                         kernel_size=4,
                                         strides=1,
                                         padding='valid',
                                         use_bias=False,
                                         name='conv')
                output = tf.nn.sigmoid(logit) 
    return output, logit

def sample_z(dim_z, num_batch):
    samples=np.random.normal(0, 1, size=[num_batch,1,1, dim_z])
    pass
    return samples

def get_losses(d_real_logits, d_fake_logits):
    
    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,labels=tf.ones_like(d_real_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,labels=tf.zeros_like(d_fake_logits)))
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,labels=tf.ones_like(d_fake_logits)))

    pass
    return d_loss, g_loss

def get_optimizers(learning_rate, beta1, beta2):
    d_optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2) # D Train step
    g_optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2)

    pass
    return d_optimizer, g_optimizer


def optimize(d_optimizer, g_optimizer, d_loss, g_loss):
    
    g_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="generator")
    d_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="discriminator")
    
    
    g_step = g_optimizer.minimize(g_loss,var_list= g_variables)
    d_step = d_optimizer.minimize(d_loss,var_list = d_variables)
    
    pass
    return d_step, g_step

def griddify_images(image_batch, size):
    h,w = image_batch.shape[1], image_batch.shape[2]
    c = image_batch.shape[3]
    img = np.zeros((int(h*size[0]), w*size[1], c))
    for idx, im in enumerate(image_batch):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = im
    return img

def train(args):
    args.data_path = '/path/to/images'
    data_loader = Dataset(args.data_path, args.num_images, args.image_size) 
    
    # sample z:
    tf.reset_default_graph()
    Z = tf.placeholder(tf.float32, shape=[None,1,1,100])
    X = tf.placeholder(tf.float32, shape=[None,64, 64,3])
    
    d_fake,d_fake_logits = generator(Z,args,reuse=tf.AUTO_REUSE) #z node outputs output & logit
    
    d_real, d_real_logits = discriminator(X,args,tf.AUTO_REUSE) #
    
    d_fake, d_fake_logits = discriminator(d_fake,args,reuse=tf.AUTO_REUSE)
    
    d_loss,g_loss = get_losses(d_real_logits, d_fake_logits)
    
    d_optimizer, g_optimizer = get_optimizers(args.lr, args.beta1, args.beta2)
    
    d_step,g_step = optimize(d_optimizer, g_optimizer, d_loss, g_loss)
    
    
    # get real im..discr(x)
    # get fake..disc(z)
    # inputs for functions
    with tf.Session() as sess:
        weights_saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        count = 0
        for epoch in range(args.n_epoch):
            print("epoch:")
            print(epoch)
            for itr, real_batch in enumerate(data_loader.get_nextbatch(args.batch_size)):
                
                
                _ = sess.run(d_step,feed_dict={Z: sample_z(args.dim_z,args.batch_size), X: real_batch}) # d_step
                _ = sess.run(g_step,feed_dict={Z: sample_z(args.dim_z,args.batch_size)}) # g_step
                fake_pic = sess.run(d_fake,feed_dict={Z: sample_z(args.dim_z,args.batch)})


                # compute losses
                if itr % 100 == 0:
                    count +=1
                    #print('epoch: ' + str(epoch) + 'iter' + str(itr))
                    d_l_i = d_loss.eval({X: real_batch,Z:sample_z(args.dim_z,args.batch_size)})
                    g_l_i = g_loss.eval({Z:sample_z(args.dim_z,args.batch_size)})
                    print(d_l_i,g_l_i)
                    

            
            if epoch == 23 and itr==3000:
                
                
                grid_images = griddify_images(fake_pic[:16], [4, 4])
                plt.savefig(grid_images,'/path/for/grid/gridded_image.png')
                #show_image(self, grid_images, normalized=True)
                print("end of code")
                #image = Image.open(grid_images)
                #image.show()
                #print("end of show")
                
                
                
            
            
                
            
            #images = sess.run([d_fake],feed_dict = {x: real_batch,Z:sample_z(args.z_dim,args.num_batch)} # returns fake images
            # save 16 images
            # choose first 16 images put on grid write function for this
            #images = images[1:16]
            # put them on grid
   
            
            
train(args) 
