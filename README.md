# Face Recognisation using Transfer Learning with VGG16 model


## Transfer Learning


Transfer learning (TL) is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although formal ties between the two fields are limited. From the practical standpoint, reusing or transferring information from previously learned tasks for the learning of new tasks has the potential to significantly improve the sample efficiency of a reinforcement learning agent.

<img src="https://www.sai-tai.com/blog/wp-content/uploads/2017/04/transfer_learningvgg16.png" width=600 height=300 align="center">


### ARCHITECTURE OF VGG16 MODEL     
   
   
   <img src="https://engmrk.com/wp-content/uploads/2018/10/VGG16_Summary-Table.jpg" width=600 height=300 align="center">
   
   
   
  ##### DATA COLLECTION USING DATACOLLECTOR.PY CODE
   
```python    
import os 
import cv2

def photo(maxcount):
  '''maxcount means how many photos you want to click'''

    folder=r"foldername" 

    #for example : folder for testing:
    # r"D:/data/train/foldername/"
    # r""D:/data/test/foldername/"
    
    
    cap=cv2.VideoCapture(0)
    count=0
    while True:
        status,image=cap.read()
        if status:
            count=count+1
            image=image[100:500,200:700]
            file="file{0}".format(count) + ".jpg"
            file=os.path.join(folder,file)
            cv2.imwrite(file,image)
            cv2.putText(image,str(count),(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,234,0),2)
            cv2.imshow("photo",image)

        
        
            if cv2.waitKey(100) == ord("q") or count == int(max_count): 
                break
    cv2.destroyAllWindows()
    print("Total photo clicked {0}".format(count))
    cap.release()
         
         
         ```
  
  
   

