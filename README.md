# Position list combined with deep autofocus


For any experiment requiring overnight/24 hrs observation, e.g., drug treatment and changing of cell phase, it is essential to keep the objects at focus. However, any movement of the stage will cause the drift on the axial direction, i.e., defocused. In this work, the long-term observation was achieved by combining the works in https://github.com/tehsinchen/Deep_Autofocus and https://github.com/tehsinchen/Position_List_PIStage.


## Results

Below is the video that recorded the full operation of going to the preset destinations in position list, centralizing the cell and finding the focal plane. The process of centralizing was speeded up 15 times, 0.8x for finding focus and 1.2x for going to the next position. Further, the centeralization was done by using the predicted mask in this project: https://github.com/tehsinchen/Eff-Unet-Keras to locate the cell.

![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/operation_demo/position_list_autofocus.gif)




For the option of non-fixed position which means the lateral and axial positions were all updated by machine, here is the results of snapshot images for 11 cycles:

![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/non-fixed-position/Pos1.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/non-fixed-position/Pos2.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/non-fixed-position/Pos3.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/non-fixed-position/Pos4.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/non-fixed-position/Pos5.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/non-fixed-position/Pos6.gif)




For the option of fixing lateral-position which can be applied for watching the movement of cells (axial direction was still determined by machine), here is the results:

![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/fixed-position/Pos1.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/fixed-position/Pos2.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/fixed-position/Pos3.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/fixed-position/Pos4.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/fixed-position/Pos5.gif)
![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/fixed-position/Pos6.gif)


![image](https://github.com/tehsinchen/Deep_Position_List/blob/main/axial_positions/axial_position.png)
