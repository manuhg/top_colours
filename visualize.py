import os
import sys
import PIL
from io import BytesIO
from six.moves import urllib
import numpy as np
from PIL import Image
import json
import cv2

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
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e,exc_tb.tb_lineno)
    return

def visualize_json(json_file):
  try:
    with open(json_file,'r') as f:    
      data = json.load(f)
      for k in data.keys():
        impose_json_on_image(data[k])
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e,exc_tb.tb_lineno,json_file)

def main():
  opf = ''
  if len(sys.argv)<2:
    print('Assuming json file to be output.json')
    opf = 'output.json'
  else:
    opf = sys.argv[1]
  visualize_json(opf)

if __name__=="__main__":
  main()