import imageio
import glob

images = []
folder_name = 'images/all_segmentation'
filenames = glob.glob('{}/*.png'.format(folder_name))
# filenames = [folder_name + '/{0:4d}.png'.format(i) for i in range(10, 100, 10)]

for filename in filenames[:100]:
    images.append(imageio.imread(filename))
imageio.mimsave('images/segmentation.gif', images, format='GIF-FI', fps=2)
