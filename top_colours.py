import os
import sys
import operator
import PIL
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from functools import reduce
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy import spatial
# import Levenshtein as leven
import json
import cv2

try:
    global webcolors
    import webcolors
except Exception as e:
    os.popen('pip install webcolors').read()


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


def run_visualization(url, MODEL):
    """Inferences DeepLab model and visualizes result."""
    try:
        f = urllib.request.urlopen(url)
        jpeg_str = f.read()
        original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check url: ' + url)
        return

    print('Image %s' % url)
    resized_im, seg_map = MODEL.run(original_im)
#   vis_segmentation(resized_im, seg_map)
    return resized_im, seg_map


def deeplab_init():
    MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
    # ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = 'deeplab_model.tar.gz'

    model_dir = tempfile.mkdtemp()
    tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    print('downloading model, this might take a while...')
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                               download_path)
    print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')
    return MODEL

def sd(a,b,nbits=8):
  return (a<<nbits)+b

def append_numbers(arr):
  return reduce(sd,arr)

def unappend_numbers(num):
  return list(map(lambda v:v&0xff,[num>>16,num>>8,num]))

def colour_strip(colour_pixel_rgb,shape=(30,120)):
  y=np.array(colour_pixel_rgb,dtype='uint8').reshape(1,-1)
  return PIL.Image.fromarray(np.full((shape[0],shape[1],3),y))

def inc_dict(dct,key):
  dct[key] = dct.get(key,0)+1

def get_dominant_obj_code(dct):
  return list(map(lambda i: i[0],sorted(dct.items(), key=operator.itemgetter(1),reverse=True)))

def dominant_obj_colours(img,mask,hist,bcg_obj=0,bcg_colour=[0,0,0]): #0 is black
  colours = {}
  bcg_colour = np.array(bcg_colour)
  img_arr = np.array(img)
  hc = hist.copy()
  hc[bcg_obj]=0
  dom_obj_data =[]
  dominant_obj_code = get_dominant_obj_code(hc)[0]
  nrows,ncols = mask.shape
  for row in range(nrows):
    for col in range(ncols):
      if mask[row,col]==dominant_obj_code:
        pix = img_arr[row,col]
        index = append_numbers(pix)
        colours[index] = colours.get(index,0)+1
        dom_obj_data.append(pix)
  return colours,dom_obj_data

def dom_colour(colours):
  dct = colours
  dc = sorted(dct.items(), key=operator.itemgetter(1),reverse=True)[0]
  dominant_colour = {'hex':dc[0],'rgb':unappend_numbers(dc[0])}
  print 'Most dominant colour','#'+str(dominant_colour['hex'])
  return colour_strip(dominant_colour['rgb'])

def dom_colour_range(dom_obj_data,n_clusters=10):
  km = KMeans(n_clusters=n_clusters)
  km.fit(dom_obj_data)

  cluster_num = list(map(int,km.predict(dom_obj_data)))
  cluster_sizes = list(map(cluster_num.count,set(cluster_num)))
  cluster_centers = list(km.cluster_centers_)
  data = dict(zip(cluster_sizes,cluster_centers))
  
  sk = data.keys()
  sk.sort(reverse=True)
  cl_centers = [ data[e] for e in sk]
  
  print 'Most dominant colour range','#'+hex(append_numbers(list(map(int,cl_centers[0]) )))
  return colour_strip(cl_centers[0])

def colour_ranges(dom_obj_data,n_clusters=10,echo=False):
  km = KMeans(n_clusters=n_clusters)
  km.fit(dom_obj_data)

  cluster_num = list(map(int,km.predict(dom_obj_data)))
  cluster_sizes = list(map(cluster_num.count,set(cluster_num)))
    
  cluster_centers = list(km.cluster_centers_)
  s = float(sum(cluster_sizes))
  percentages = list(map(lambda v:int(round((100*v)/s)),cluster_sizes))
  
  data = pd.DataFrame({'cluster_sizes':cluster_sizes,'cluster_centers':cluster_centers,'percentages':percentages})
  data.sort_values(by=['cluster_sizes'],ascending=False)
  cluster_sizes = list(data.loc[:,'cluster_sizes'])
  cl_centers = list(data.loc[:,'cluster_centers'])
  percentages = list(data.loc[:,'percentages'])
  
  cl_centers = list(map(lambda e: list(map(int,e)),cl_centers))
  if echo:
    print 'Ratio of colour ranges',ratio
  
  
  images = list(map(colour_strip,cl_centers))
  widths, heights = zip(*(i.size for i in images))

  max_width = max(widths)
  total_height = sum(heights)

  new_im = Image.new('RGB', (max_width, total_height))

  y_offset = 0
  if echo:
    print 'colour ranges sorted by dominance. (High to Low)'
    
  for im in images:
    new_im.paste(im, (0,y_offset))
    y_offset += im.size[1]
  return new_im,percentages,cl_centers


def dstsq(a,b,i):
  return (a[i]-b[i])**2

def euclidean(hex1,hex2):
  c1,c2 = list(map(unappend_numbers,[hex1,hex2]))
  return sum(list(map( lambda i: dstsq(c1,c2,i),list(range(0,3)))))

def rgb2hexstr(rgb_val):
  return '#'+hex(append_numbers(rgb_val)).replace('0x','').rjust(6,'0')

def get_colour_name(colour_rgb):
  valindex = colours_tree.query(colour_rgb)[1]
  rgb_val = web_colours_rgb[valindex]
  hexcode = rgb2hexstr(rgb_val)
  #hexcode = hexcode.replace('L','')
  colour_name = webcolors.css3_hex_to_names[hexcode]
  for k,v in replacements.items():
    colour_name = colour_name.replace(k,v)
  return str(colour_name)

def get_top_dominant_colours(image_url,Model):
  hist = {}
  image,mask = run_visualization(image_url,Model)
  n_clusters=3

  list(map( lambda imrow: list(map(lambda k: inc_dict(hist,k),imrow)),mask))
  colours,dom_obj_data = dominant_obj_colours(image,mask,hist)

  _,percentages,cl_centers = colour_ranges(dom_obj_data,n_clusters)
  colour_names = list(map(get_colour_name,cl_centers))
  
  cl_centers_hex = list(map(rgb2hexstr,cl_centers))
  return cl_centers_hex,colour_names,percentages

def get_top_dominant_colours_json(image_data,Model):
  image_id,image_url,image_description = image_data
  cl_centers_hex,colour_names,percentages = get_top_dominant_colours(image_url,Model)
  json_data = {'image':image_url,
               'description':image_description,
               'id':image_id}
  
  for i in range(len(cl_centers_hex)):
    cdata = {'colour-'+str(i+1):{'name':colour_names[i],
                                 'hex':cl_centers_hex[i],
                                 'percentage':percentages[i]}}
    json_data.update(cdata)
  
  return json_data

def process_file(input_filename,output_filename='output.json'):
  global web_colours_rgb,colours_tree,replacements
  web_colours_rgb = list(map(unappend_numbers,list(map(lambda v:int(v.replace('#','0x'),0),webcolors.css3_hex_to_names.keys()))))
  colours_tree = spatial.KDTree(web_colours_rgb)
  replacements = {'light':'light ','dark':'dark ','dim':'dark '}

  Model = deeplab_init()

  try:
    file = open(input_filename,'r')
    data = []
    for line in file:
      dt = line.split('|')
      if len(dt)>1:
        dt = dt[1:]
        data.append(dt)
    file.close()
    
    jsons = list(map(lambda d:get_top_dominant_colours_json(d,Model),data))
    final_data = {}
    for j in jsons:
      final_data.update({j['id']:j})
    
    print(final_data)
    
    with open(output_filename,'w') as of:
      json.dump(final_data,of)
    
  except Exception as e:
    print(e)
    
def main():
  if len(sys.argv)<2:
    print('No input file specified!')
    return
  
  input_file = sys.argv[1]
  output_file = 'output.json'
  if len(sys.argv)>2:
    output_file = sys.argv[2]
  
  process_file(input_file,output_file)
  
if __name__ == "__main__":
  main()