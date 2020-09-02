from flask import Flask,render_template,session,url_for,make_response
from flask import request, redirect
from werkzeug.utils import secure_filename
import os
import random
from functools import wraps, update_wrapper
from datetime import datetime
from collections import Counter
#machine learning libraries


import random
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from matplotlib.image import imread
import matplotlib.pyplot as plt
import PIL
import cv2
from PIL import Image
import matplotlib
import scipy.ndimage


def resize(image,window_height):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height/aspect_ratio
    image = cv2.resize(image, (int(window_height),int(window_width)))
    return image


def clustering(k,path,code,file):
    img_r = cv2.imread(path)
    #print(img_r.shape)
    #print('inside function ',img_r.shape[2])
    #print(code + file.split('.')[0])
    if img_r.shape[2] > 2 and img_r.shape[2] <= 4:
      img=resize(img_r,500)
      cv2.imwrite('static/png_process/'+code+file.split('.')[0]+'.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY), 80])
      path= 'static/png_process/'+code+file.split('.')[0]+'.jpg'
    img = imread(path)



    X = img.reshape(-1, img.shape[2])


    #print("%d kb before" % (X.size * X.itemsize))
    #print(X.shape)
    X = X / 255
    #print("%d kb after" % (X.size * X.itemsize))
    #print(img)
    #plt.imshow(X)
    kmeans = KMeans(n_clusters=k).fit(X)
    #print(kmeans.cluster_centers_)
    #print(kmeans.labels_.shape)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    #print('labels',segmented_img.shape)
    #g=[1,1,0,2]

    #counts = list(set(kmeans.labels_))


    #center_colors = kmeans.cluster_centers_
    # We get ordered colors by iterating through the keys
    #ordered_colors = [center_colors[i] for i in counts]
    #print('lllll',ordered_colors)

    #rgb_colors = [ordered_colors[i] for i in counts]
    #print('test',rgb_colors)
    #hex_colors = [RGB2HEX(rgb_colors[i]) for i in counts.keys()]
    #plt.figure(figsize=(8, 6))
    #value_list=[5 for i in range(0,k)]
    #plt.pie(value_list,colors=ordered_colors)

    #pie_path = 'static/pie/' + 'converted' + code + file
    #plt.savefig(pie_path)
    #print(kmeans.labels_)
    #segmented_img =kmeans.labels_
    #print(kmeans.cluster_centers_)
    #print('xxxxxxxx',kmeans.cluster_centers_[g])
    #print(kmeans.labels_)
    #print(segmented_img.shape,img.shape)
    segmented_img = segmented_img.reshape(img.shape)
    #print("%d kb after clustering" % (segmented_img.size * segmented_img.itemsize))
    #print(img.shape)
    #plt.figure(figsize=(12, 12))
    #plt.imshow(segmented_img)
    #plt.show()

    path='static/uploads/'+'converted'+code+file

    matplotlib.image.imsave(path, segmented_img)

app=Flask(__name__)
app.config["CACHE_TYPE"] = "null"
UPLOAD_FOLDER =  'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 50 * 1024 * 1024




def get_size(fobj):
    if fobj.content_length:
        return fobj.content_length

    try:
        pos = fobj.tell()
        fobj.seek(0, 2)  #seek to end
        size = fobj.tell()
        fobj.seek(pos)  # back to original position
        return size
    except (AttributeError, IOError):
        pass


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False




@app.route("/", methods=["GET", "POST"])
def index3():


    return render_template("k_means_demo.html")

@app.route("/about_me", methods=["GET", "POST"])
def aboutme():


    return render_template("aboutme.html")

@app.route('/display_output1')
def upload_file(k,path,code,filename):
    return none




@app.route("/display_output", methods=["GET", "POST"])
def upload_image():
  try:
    if request.method == "POST":

        print('first check')
        #print('inside post')
        f = request.files['image']
        text = request.form['text']
        if int(text)%1==0 or int(text)%1<=30:


         if get_size(f) > 10 * (1024 ** 2):
                abort(413)
                return render_template("error.html")
         else:
                code=str(random.randint(1,1000000000))
                file=f.filename

                filename=code + f.filename
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                clustering(int(text),os.path.join(app.config['UPLOAD_FOLDER'], filename),code,file)
                print(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                path_output = 'converted'+filename

                png_jpg_path = ''
                if str.upper(file.split('.')[1]) == 'PNG':
                    png_jpg_path = code + file.split('.')[0] + '.jpg'
                else:
                    png_jpg_path = filename

                return render_template("output_display.html", path_img=path_output,orig_img=png_jpg_path,shetch=path_output, clusters=int(text))

        else:
            return render_template("error.html")

  except:
      return render_template("error.html")


if __name__=='__main__':
    app.run()


