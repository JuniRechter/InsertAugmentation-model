# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:29:07 2024

@author: Juniper Rechter

Planned pseudo-code for inserting animal mask crops into empty CT images.
"""

#%%
ImageDataGenerator: feed in full df + crops df + empties df 
Read df[kfold grouping'] 
Unique list  

Open empty df, randomly select images that have grouping in unique list,  
If night = true,  
Open crops df and look only at night=true 
Randomly grab crop and insert 
If night = False,  
Open crops df and look only at night=False 
Randomly grab crop and insert 

def __cut_paste_input(self, path):
    image = tf.keras.utils.load_img(path)
    image_arr = tf.keras.utils.img_to_array(image)
    image_arr = small_aug.augment_image(image_arr.astype("uint8")).astype("float32")
    return image_arr/255.
