import cv2 as cv
import numpy as np
import utils

dest_folder = 'imgs/second_task/samples/'
samples = []
_, rgb_img_filenames = utils.get_img_filenames('second_task',start_dir='.')
print(rgb_img_filenames)

rgb_img = []
for i in range(len(rgb_img_filenames)):
    rgb_img.append(cv.cvtColor(cv.imread(rgb_img_filenames[i]),cv.COLOR_BGR2RGB))

'''samples.append(rgb_img[0][190:210,140:160])
samples.append(rgb_img[0][51:54,152:158])
samples.append(rgb_img[0][29:49,116:133])
samples.append(rgb_img[0][26:36,120:125])
samples.append(rgb_img[0][190:215,120:160])
samples.append(rgb_img[0][50:55,153:159])
samples.append(rgb_img[0][43:56,130:136])
samples.append(rgb_img[1][60:100,100:140]) #
samples.append(rgb_img[1][120:140,150:175]) ###
samples.append(rgb_img[1][140:155,50:70])
samples.append(rgb_img[1][125:135,155:177])
samples.append(rgb_img[1][28:44,110:133])
samples.append(rgb_img[1][85:135,120:170])'''

# IMG 1
samples.append(rgb_img[0][50:55,153:158]) # macchia scura in alto a destra
samples.append(rgb_img[0][41:57,129:134]) # macchia piu chiara in alto. presa in verticale
samples.append(rgb_img[0][208:218,159:168]) # macchia scura intorno al gambo, basso dx
samples.append(rgb_img[0][193:204,136:154]) # macchia scura intorno al gambo, un po piu su
samples.append(rgb_img[0][184:200,86:100]) # macchia scura intorno al gambo, a sinsitra del gambo
samples.append(rgb_img[0][190:214,123:157]) # zona scura intorno al gambo
# IMG 2
samples.append(rgb_img[1][17:39,111:135]) # in alto
samples.append(rgb_img[1][68:110,67:122]) # prende due macchiette piu scure
samples.append(rgb_img[1][70:85,67:167]) # largo, taglia a mezzo il russet
#samples.append(rgb_img[1][161:171,38:51]) # ancora piu in basso a sinistra
samples.append(rgb_img[1][83:111,139:169]) # a destra
samples.append(rgb_img[1][49:138,95:130]) # taglia in verticale
samples.append(rgb_img[1][62:93,140:168]) # a destra

for i in range(len(samples)):
    cv.imwrite('{}/sample_{}.png'.format(dest_folder,i),cv.cvtColor(samples[i],cv.COLOR_RGB2BGR))
print('ok')




