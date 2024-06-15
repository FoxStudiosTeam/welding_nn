from ultralytics import YOLO


model = YOLO("C:\\prog\\neural\\runs\\detect\\train\\weights\\best.pt")
if __name__ != '__main__': exit()

img = 'datasets/evaluate/7 (102).jpg'

results = model(img)
for result in results:
    result.show()
    


#model.train(data='data.yaml', epochs=10, batch=16, imgsz=640)
    

#model.val()

#results = model('/home/kaiv/neural/datasets/train/12 (82).jpg')  # results list

#for r in results:
    
#    im_array = r.plot()  # plot a BGR numpy array of predictions
#    r.save_txt("results.txt")
#    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#    #im.show()  # show image
#    im.save('results.jpg')  # save image
