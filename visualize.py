import os
import sys
import PIL
from io import BytesIO
from six.moves import urllib
import numpy as np
from PIL import Image
import json
import cv2
import glob

def unappend_numbers(num):
  return list(map(lambda v:v&0xff,[num>>16,num>>8,num]))

def impose_json_on_image(json_data):
  try:
    url = json_data['image']
    colours = json_data['colour-1'],json_data['colour-2'],json_data['colour-3']
    strs_to_write = list(map(lambda c:str(c['name'])+'('+str(c['percentage'])+')',colours))
  
    print(url)
    fa = urllib.request.urlopen(url)
    jpeg_str = fa.read()
    original_im = Image.open(BytesIO(jpeg_str))
    img = cv2.cvtColor(np.array(original_im), cv2.COLOR_RGB2BGR)
    for i in range(len(strs_to_write)):
      c_rgb = tuple(unappend_numbers(int(colours[i]['hex'].replace('#','0x'),0)))
      cv2.rectangle(img,(0,10*i), (10,(10*i)+10), c_rgb, cv2.FILLED)
      cv2.putText(img, strs_to_write[i], (10, (i*10)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(json_data['id']+'.jpg',img)
    fa.close()
    return img,str(json_data['id'])
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e,exc_tb.tb_lineno)
    return None,''

def visualize_json(json_file):
  try:
    with open(json_file,'r') as f:    
      data = json.load(f)
      return impose_json_on_image(data)
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e,exc_tb.tb_lineno,json_file)

def main():
  opd = ''
  if len(sys.argv)<2:
    print('Assuming json files to be in directory named output')
    opd = 'output'
  else:
    opd = sys.argv[1]
  files = list(glob.glob(opd+'/*.json'))
  images_data = list(map(visualize_json,files))
  i=0
  while(True):
    cv2.imshow(images_data[i][1],images_data[i][0])
    key = cv2.waitKey(0)
    if key == 27:
      break
    elif key == ord('<'):
      i = i+1
    elif key == ord('>'):
      i = i-1
    i = 0 if i<0 else i
  cv2.destroyAllWindows()

if __name__=="__main__":
  main()    