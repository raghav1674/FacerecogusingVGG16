#without using Haarcascade , the person need to look at the webcam for better dataset


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
