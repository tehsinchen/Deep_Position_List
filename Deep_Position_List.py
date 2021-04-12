import os
import glob
import tkinter as tk
from tkinter import filedialog
import threading

import numpy as np
from numpy import linalg as LA
from tensorflow.keras.models import load_model
import tifffile

from PyDAQmx import Task
import PyDAQmx as Daq
import time

from pipython import GCSDevice, pitools
from PhPy import phGetPh, phDoPh, phSetPh
from PhPy import phGetCam, phDoCam, phSetCam
from PhPy import phGetCine, phDoCine, phSetCine

PhantomDictionary = phGetPh()
globals().update(PhantomDictionary)


class ThreadBG(threading.Thread):

    def __init__(self, state, interval=None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.state = state
        self.interval = interval
        self.start()

    def run(self):
        task = Task()
        port = "Dev1/port1/line0"
        pulse = np.zeros(1, dtype=np.uint8)
        task.CreateDOChan(port, "", Daq.DAQmx_Val_ChanPerLine)
        task.StartTask()
        pulse[0] = self.state
        task.WriteDigitalLines(1, False, Daq.DAQmx_Val_WaitInfinitely,
                               Daq.DAQmx_Val_GroupByChannel,
                               pulse, None, None)
        if self.interval is not None:
            time.sleep(self.interval)
            pulse[0] = 0
            task.WriteDigitalLines(1, False, Daq.DAQmx_Val_WaitInfinitely,
                                   Daq.DAQmx_Val_GroupByChannel,
                                   pulse, None, None)
        task.StopTask()
        task.ClearTask()


class Utils:

    @staticmethod
    def last_num(file_dir):
        seq_list = []
        dic_file = {}
        try:
            for r in range(len(file_dir)):
                dic_file[file_dir[r]] = int(file_dir[r].split("\\")[-1].split(".")[0].split('_')[-1])
            file_directory = sorted(dic_file.items(), key=lambda x: x[1])
            for file in file_directory:
                seq_list.append(file[1])
            return int(seq_list[-1])
        except IndexError:
            return 0

    def ph_save(self, filename, cur_pos):
        cur_pos = cur_pos.strip()
        def ProgressIndicator(hcine, percent):
            #print("Cine save: hcine = ", hcine, "percent = ", percent,
            #      "%  =======================================================")
            if percent == 100:
                phDoCine(Close, hcine)
            return 1

        cn = 0
        cur_cine = phGetCam(ActivePartition, cn)[0]
        if cur_cine == 0:
            phDoCam(Record, cn)
            cur_cine += 1
        phDoCam(Trigger, cn)
        while phGetCam(Recorded, cn, cur_cine) != 1:
            pass
        hcine = phGetCam(CineHandle, cn, cur_cine)
        if filename.find('.') != -1:
            filename = filename.split('.')[0]
        directory = os.path.dirname(filename)
        result_dir = os.path.join(directory, cur_pos)
        os.makedirs(result_dir, exist_ok=True)
        file_name = os.path.basename(filename)
        file_list = glob.glob(os.path.join(result_dir, "*.cine"))
        file_index = int(self.last_num(file_list) + 1)
        file_path = os.path.join(result_dir, f'{file_name}_{file_index}.cine')
        phSetCine(SaveName, hcine, file_path)
        phSetCine(SaveType, hcine, svv_RawCine)
        (f, l) = phGetCine(Range, hcine)
        phSetCine(SaveRange, hcine, (f, l))
        phSetPh(Callback0, ProgressIndicator)
        phDoCine(Save, hcine)

    def snap_img(self, filename, cur_pos):

        directory = os.path.dirname(filename)
        result_dir = os.path.join(directory, cur_pos)
        os.makedirs(result_dir, exist_ok=True)
        file_name = os.path.basename(filename)
        file_list = glob.glob(os.path.join(result_dir, "*.tif"))
        file_index = int(self.last_num(file_list) + 1)
        file_path = os.path.join(result_dir, f'{file_name}_{file_index}.tif')

        hcine = phGetCam(CineHandle, 0, -1)
        image = phGetCine(Image_np, hcine, (0, 0, gci_LeftAlign))[0]
        phDoCine(Close, hcine)
        tifffile.imsave(file_path, image)


    def img_preprocessing(self, arr, nor=None, new_shape=None):
        if nor is not None:
            arr = arr / nor
            if new_shape is not None:
                if len(arr.shape) == 3:
                    arr_out = np.ndarray((arr.shape[0], new_shape[0], new_shape[1]))
                    for i in range(arr.shape[0]):
                        arr_out[i, :, :] = self.binning(arr[i, :, :], new_shape)
                else:
                    arr_out = self.binning(arr, new_shape)
            else:
                arr_out = arr
        else:
            pass
        return arr_out

    @staticmethod
    def binning(arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)

    @staticmethod
    def find_min_distance(image, cen_loc, edge_value):
        indices = np.argwhere(image < edge_value)
        if len(indices) == 0:
            return None
        else:
            m = float('inf')
            for index in indices:
                index = np.array([index[0] + cen_loc[0] - 1, index[1] + cen_loc[1] - 1])
                distance = LA.norm((cen_loc - index), 2)
                m = min(m, distance)
            return m

    def search_min(self, image, cur_loc, edge_value):
        space = 10
        img_x = image.shape[0]
        img_y = image.shape[1]
        distance = None
        while distance is None:
            x1 = self.get_valid_index(cur_loc[0] - space, img_x)
            x2 = self.get_valid_index(cur_loc[0] + space, img_x)
            y1 = self.get_valid_index(cur_loc[1] - space, img_y)
            y2 = self.get_valid_index(cur_loc[1] + space, img_y)
            img = image[x1:x2, y1:y2]
            distance = self.find_min_distance(img, cur_loc, edge_value)
            space += 5
        return distance

    @staticmethod
    def get_valid_index(x, shape):
        if x > shape:
            return shape + 1
        elif x < 0:
            return 0
        else:
            return x

    def find_center(self, img, edge_value):
        portion = 0
        largest = 0
        center = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > edge_value:
                    portion += 1
                    pos = [i, j]
                    distance = self.search_min(img, pos, edge_value)
                    if distance > largest:
                        largest = distance
                        center = pos
        if portion > 1000:
            return center
        else:
            return None

    def find_edge(self, image, edge_value):
        center = self.find_center(image, edge_value)
        edge = {}
        try:
            arr_vertical = image[:, center[1]]
            arr_horizontal = image[center[0], :]
            for i in range(len(arr_vertical)):
                try:
                    up = arr_vertical[center[0] - i]
                    down = arr_vertical[center[0] + i]
                    left = arr_horizontal[center[1] - i]
                    right = arr_horizontal[center[1] + i]
                    if up < 0.5:
                        if 'up' not in edge:
                            edge['up'] = center[0] - i + 1
                    if down < 0.5:
                        if 'down' not in edge:
                            edge['down'] = 128 - (center[0] + i - 1)
                    if left < 0.5:
                        if 'left' not in edge:
                            edge['left'] = center[1] - i + 1
                    if right < 0.5:
                        if 'right' not in edge:
                            edge['right'] = 128 - (center[1] + i - 1)
                except IndexError:
                    pass
            vertical_correction = edge['up'] - edge['down']
            horizontal_correction = edge['left'] - edge['right']
            return [vertical_correction + 63, horizontal_correction + 63]
        except:
            return None

    def centralize(self, model_lateral, e518, edge_value=0.5):
        x = e518.axes[1]
        y = e518.axes[0]
        z = e518.axes[2]
        operation = 1
        for attempt in range(5):
            current_pos = [e518.qPOS(x)[x], e518.qPOS(y)[y], e518.qPOS(z)[z]]
            hcine = phGetCam(CineHandle, 0, -1)
            image = phGetCine(Image_np, hcine, (0, 0, gci_LeftAlign))[0, 0:640, :]
            phDoCine(Close, hcine)
            image = self.img_preprocessing(arr=image, nor=65535, new_shape=(128, 128))
            mask_img = model_lateral.predict(image.reshape((1, 128, 128, 1)))[0, :, :, 0]
            if attempt < 3:
                center = self.find_center(mask_img, edge_value)
            else:
                if operation > 3:
                    center = self.find_edge(mask_img, edge_value)
                else:
                    center = None
            if center is not None:
                operation += 1
                mod_x = int(center[1] - 63) * 0.04 * 5
                mod_y = int(center[0] - 63) * 0.04 * 5
                try:
                    e518.MOV(x, current_pos[0] + mod_x)
                    e518.MOV(y, current_pos[1] + mod_y)
                    e518.MOV(z, current_pos[2])
                    pitools.waitontarget(e518)
                except:
                    e518.MOV(x, 100)
                    e518.MOV(y, 100)
                    e518.MOV(z, 100)
                    pitools.waitontarget(e518)
                    return None
        if operation == 1:
            return None
        else:
            return True

    def auto_focus(self, model_coarse, model_fine, e518):
        z = e518.axes[2]
        cur_z = e518.qPOS(z)[z]
        self.coarse_mod(z, cur_z, e518, model_coarse)
        cur_z = e518.qPOS(z)[z]
        self.fine_mod(z, cur_z, e518, model_fine)

    def coarse_mod(self, z, cur_z, e518, model_coarse):
        image_coarse = np.ndarray((3, 640, 640))
        coarse_moving_list = [-1, 0, 1]
        for i in coarse_moving_list:
            e518.MOV(z, cur_z + i)
            pitools.waitontarget(e518, axes=z)
            hcine = phGetCam(CineHandle, 0, -1)
            image_coarse[i + 1, :, :] = phGetCine(Image_np, hcine, (0, 0, gci_LeftAlign))[0, 0:640, :]
            phDoCine(Close, hcine)
        imgs_coarse = self.img_preprocessing(arr=image_coarse, nor=65535, new_shape=(320, 320))[:, 32:288, 32:288]
        imgs_coarse = np.reshape(imgs_coarse, (1, 256, 256, 3))
        c1, c2, c3 = model_coarse.predict(imgs_coarse)[0]
        coarse_key_frame = [c1, c2, c3]
        #print(f'coarse predicted result: {coarse_key_frame}')
        coarse_predicted_z = coarse_moving_list[coarse_key_frame.index(max(coarse_key_frame))]
        e518.MOV(z, cur_z + coarse_predicted_z)
        #print(cur_z + coarse_predicted_z, coarse_predicted_z)
        pitools.waitontarget(e518)
        # coarse modification

    def fine_mod(self, z, cur_z, e518, model_fine):
        image_fine = np.ndarray((7, 640, 640))
        fine_moving_list = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        for j in range(len(fine_moving_list)):
            e518.MOV(z, cur_z + fine_moving_list[j])
            pitools.waitontarget(e518, axes=z)
            hcine = phGetCam(CineHandle, 0, -1)
            image_fine[j, :, :] = phGetCine(Image_np, hcine, (0, 0, gci_LeftAlign))[0, 0:640, :]
            phDoCine(Close, hcine)
        image_fine = self.img_preprocessing(arr=image_fine, nor=65535, new_shape=(320, 320))[:, 32:288, 32:288]
        image_fine = np.reshape(image_fine, (1, 256, 256, 7))
        f1, f2, f3, f4, f5, f6, f7 = model_fine.predict(image_fine)[0]
        fine_key_frame = [f1, f2, f3, f4, f5, f6, f7]
        # print(f'fine predicted result: {fine_key_frame}')
        fine_predicted_z = fine_moving_list[fine_key_frame.index(max(fine_key_frame))]
        e518.MOV(z, cur_z + fine_predicted_z)
        # print(cur_z + fine_predicted_z, fine_predicted_z)
        pitools.waitontarget(e518)
        # fine modification

    def automatic(self, e518, model_lateral, model_axial_coarse, model_axial_fine):
        is_cell = self.centralize(model_lateral, e518)
        if is_cell:
            self.auto_focus(model_axial_coarse, model_axial_fine, e518)


class PositionList(tk.Frame):

    def __init__(self, root, c867, e518):

        self.e518 = e518
        self.x = self.e518.axes[1]
        self.y = self.e518.axes[0]
        self.z = self.e518.axes[2]
        self.c867 = c867
        self.mox = self.c867.axes[0]
        self.moy = self.c867.axes[1]

        self.popup_win = tk.Toplevel(root)
        self.popup_win.wm_title("Position list")
        self.popup_win.geometry('700x800')
        self.popup_win.configure(bg='white')
        self.popup_win.resizable(0, 0)

        self.absolute_btn = tk.Button(self.popup_win,
                                      text='Absolute Position',
                                      font='Arial 12',
                                      width=31,
                                      command=self.set_absolute_pos)
        self.absolute_btn.place(relx=0.05, rely=0.02)
        self.relative_btn = tk.Button(self.popup_win,
                                      text='Relative Position',
                                      font='Arial 12',
                                      width=31,
                                      command=self.set_relative_pos)
        self.relative_btn.place(relx=0.52, rely=0.02)
        interval = 0.068
        self.add_btn = tk.Button(self.popup_win,
                                 text='Add',
                                 font='Arial 12',
                                 width=12,
                                 command=self.add_position)
        self.add_btn.place(relx=0.79, rely=0.112)

        self.remove_btn = tk.Button(self.popup_win,
                                    text='Remove',
                                    font='Arial 12',
                                    width=12,
                                    command=self.remove_position)
        self.remove_btn.place(relx=0.79, rely=0.112+interval)

        self.clean_btn = tk.Button(self.popup_win,
                                   text='Clean',
                                   font='Arial 12',
                                   width=12,
                                   command=self.clean_position)
        self.clean_btn.place(relx=0.79, rely=0.112+(interval*2))

        self.insert_btn = tk.Button(self.popup_win,
                                    text='Insert',
                                    font='Arial 12',
                                    width=12,
                                    command=self.insert_position)
        self.insert_btn.place(relx=0.79, rely=0.112+(interval*3))

        self.replace_btn = tk.Button(self.popup_win,
                                     text='Replace',
                                     font='Arial 12',
                                     width=12,
                                     command=self.replace_position)
        self.replace_btn.place(relx=0.79, rely=0.112+(interval*4))

        self.update_btn = tk.Button(self.popup_win,
                                    text='Update',
                                    font='Arial 12',
                                    width=12,
                                    command=self.update_position)
        self.update_btn.place(relx=0.79, rely=0.112+(interval*5))

        self.go_btn = tk.Button(self.popup_win,
                                text='Go',
                                font='Arial 12',
                                width=12,
                                command=self.go)
        self.go_btn.place(relx=0.79, rely=0.112+(interval*6))

        self.start_btn = tk.Button(self.popup_win,
                                   text='Start',
                                   font='Arial 12',
                                   width=12,
                                   command=self.start)
        self.start_btn.place(relx=0.79, rely=0.112+(interval*7))

        self.save_btn = tk.Button(self.popup_win,
                                  text='Save',
                                  font='Arial 12',
                                  width=12,
                                  command=self.save)
        self.save_btn.place(relx=0.79, rely=0.112+(interval*8)+0.023)

        self.load_btn = tk.Button(self.popup_win,
                                  text='Load',
                                  font='Arial 12',
                                  width=12,
                                  command=self.load)
        self.load_btn.place(relx=0.79, rely=0.112+(interval*9)+0.023)

        self.close_btn = tk.Button(self.popup_win,
                                   text='Close',
                                   font='Arial 12',
                                   width=12,
                                   command=root.destroy)
        self.close_btn.place(relx=0.79, rely=0.112+(interval*10)+0.045)

        self.period_label = tk.Label(self.popup_win,
                                     text='Interval (min):',
                                     font='Arial 12', bg='white')
        self.period_label.place(relx=0.02, rely=0.91)
        self.period = tk.Entry(self.popup_win, font='Arial 12', width=5)
        self.period.place(relx=0.17, rely=0.91)

        self.nb_loop_label = tk.Label(self.popup_win,
                                      text='Number of loops:',
                                      font='Arial 12', bg='white')
        self.nb_loop_label.place(relx=0.27, rely=0.91)
        self.nb_loop = tk.Entry(self.popup_win, font='Arial 12', width=5)
        self.nb_loop.place(relx=0.46, rely=0.91)
        self.nb = 1

        self.shutter_label = tk.Label(self.popup_win, text='Shutter interval (s):',
                                       font='Arial 12', bg='white')
        self.shutter_label.place(relx=0.02, rely=0.96)
        self.shutter = tk.Entry(self.popup_win, font='Arial 12', width=5)
        self.shutter.place(relx=0.22, rely=0.96)

        self.filepath_label = tk.Label(self.popup_win, text='File path:',
                                       font='Arial 12', bg='white')
        self.filepath_label.place(relx=0.3, rely=0.96)
        self.filepath = tk.Entry(self.popup_win, font='Arial 12', width=35)
        self.filepath.place(relx=0.4, rely=0.96)

        self.ref_pos_label = tk.Label(self.popup_win, text='Ref. pos:',
                                      font='Arial 12', bg='white')
        self.ref_pos_label.place(relx=0.015, rely=0.07)
        self.pre_ref = ''
        self.cur_ref = ''
        self.ref_pos_list = []
        self.ref_position = tk.StringVar()
        self.ref_pos = tk.Entry(self.popup_win, font='Arial 12', width=38, textvariable=self.ref_position)
        self.ref_pos.place(relx=0.12, rely=0.074)
        self.set_ref = tk.Button(self.popup_win,
                                 text='Set ref.',
                                 font='Arial 12',
                                 width=8,
                                 command=self.set_ref_position)
        self.set_ref.place(relx=0.64, rely=0.07)
        self.ref_pos_label.place_forget()
        self.ref_pos.place_forget()
        self.set_ref.place_forget()

        self.position_list = []
        self.pos_list = tk.StringVar()
        self.position_listbox = tk.Listbox(self.popup_win,
                                           bg='white',
                                           font='Arial 12',
                                           height=20,
                                           listvariable=self.pos_list,
                                           selectmode=tk.SINGLE)
        self.position_listbox.place(relx=0.01, rely=0.11, relwidth=0.75, relheight=0.77)

        self.apply_var = tk.BooleanVar()
        self.apply_var.set(False)
        self.apply = tk.Checkbutton(self.popup_win,
                                    bg='white',
                                    text='Apply machine?',
                                    variable=self.apply_var,
                                    command=self.apply_machine)
        self.apply.place(relx=0.57, rely=0.905)
        self.apply.configure(font='Arial 12')

        self.fix_var = tk.BooleanVar()
        self.fix_var.set(False)
        self.fix = tk.Checkbutton(self.popup_win,
                                  bg='white',
                                  text='Fix xy?',
                                  variable=self.fix_var,
                                  command=self.fix_xy)
        self.fix.place(relx=0.8, rely=0.905)
        self.fix.configure(font='Arial 12')

        self.utils = Utils()
        self.model_lateral = []
        self.model_axial_coarse = []
        self.model_axial_fine = []

        self.z_dict = {}

    def set_absolute_pos(self):
        self.absolute_btn['relief'] = tk.SUNKEN
        self.relative_btn['relief'] = tk.RAISED
        self.ref_pos_label.place_forget()
        self.ref_pos.place_forget()
        self.set_ref.place_forget()

    def set_relative_pos(self):
        self.absolute_btn['relief'] = tk.RAISED
        self.relative_btn['relief'] = tk.SUNKEN
        self.ref_pos_label.place(relx=0.015, rely=0.07)
        self.ref_pos.place(relx=0.12, rely=0.074)
        self.set_ref.place(relx=0.64, rely=0.07)

    def set_ref_position(self):
        cur_pos = self.get_current_position()
        ref_pos = cur_pos['motor'] + cur_pos['piezo']
        self.ref_position.set(str(ref_pos))
        self.cur_ref = self.ref_position.get()

    def add_position(self):
        index = int(self.position_listbox.size())
        pos = self.get_formatted_position(index)
        self.position_list.append(pos)
        self.pos_list.set(self.position_list)

    def remove_position(self):
        item = self.position_listbox.curselection()
        if len(item) != 0:
            index = item[0]
            self.position_listbox.delete(item)
            self.position_list.pop(index)
            self.update_position_list(self.position_list)

    def clean_position(self):
        self.pos_list.set('')
        self.position_list = []

    def insert_position(self):
        item = self.position_listbox.curselection()
        if len(item) != 0:
            index = item[0]
            pos = self.get_formatted_position(index)
            self.position_listbox.insert(index, pos)
            self.position_list.insert(index, pos)
            self.update_position_list(self.position_list)

    def replace_position(self):
        item = self.position_listbox.curselection()
        if len(item) != 0:
            index = item[0]
            pos = self.get_formatted_position(index)
            self.position_list[index] = pos
            self.pos_list.set(self.position_list)

    def replace_z(self):
        item = self.position_listbox.curselection()
        if len(item) != 0:
            index = item[0]
            original_pos = self.position_list[index]
            original_z = self.get_valid_position(original_pos)['piezo'][2]
            new_pos = self.get_formatted_position(index)
            new_z = self.get_valid_position(new_pos)['piezo'][2]
            result_pos = original_pos.replace(str(original_z), str(new_z))
            self.position_list[index] = result_pos
            self.pos_list.set(self.position_list)
            self.go()

    def record_z(self, pos_nb):
        item = self.position_listbox.curselection()
        if len(item) != 0:
            index = item[0]
            cur_pos = self.position_list[index]
            cur_z = self.get_valid_position(cur_pos)['piezo'][2]
            if str(pos_nb) in self.z_dict:
                self.z_dict[str(pos_nb)].append(cur_z)
            else:
                self.z_dict[str(pos_nb)] = [cur_z]

    def update_position(self):
        if self.relative_btn['relief'] == 'sunken':
            if self.pre_ref == '':
                self.pre_ref = self.cur_ref
            self.ref_pos_list = self.position_list
            if self.cur_ref != '' and len(self.ref_pos_list) != 0:
                pre_pos = eval(self.pre_ref)
                cur_pos = eval(self.cur_ref)
                diff_pos = [pre-cur for pre, cur in zip(pre_pos, cur_pos)]
                update_position_list = []
                for i, pos in enumerate(self.ref_pos_list):
                    pos_dic = self.get_valid_position(pos)
                    pos_list = pos_dic['motor'] + pos_dic['piezo']
                    new_list = [float((pre-diff).__format__('.4f')) for pre, diff in zip(pos_list, diff_pos)]
                    new_pos = self.get_formatted_position(index=i, motor=new_list[:2], piezo=new_list[2:])
                    update_position_list.append(new_pos)
                self.position_list = update_position_list
                self.pos_list.set(self.position_list)

    def go(self):
        item = self.position_listbox.curselection()
        if len(item) != 0:
            self.go_btn['state'] = tk.DISABLED
            index = item[0]
            pos = self.position_list[index]
            position_dic = self.get_valid_position(pos)
            self.move(position_dic['motor'], position_dic['piezo'])
            self.go_btn['state'] = tk.NORMAL

    def start(self):
        self.position_listbox.selection_clear(0, tk.END)
        self.start_btn['state'] = tk.DISABLED
        for i, pos in enumerate(self.position_list):
            self.position_listbox.selection_set(i)
            if i == 0:
                list_length = len(self.position_list)
                if list_length != 1:
                    self.position_listbox.selection_clear(list_length-1)
            else:
                self.position_listbox.selection_clear(i-1)
            self.position_listbox.update()
            pos = self.position_list[i]
            position_dic = self.get_valid_position(pos)
            self.move(position_dic['motor'], position_dic['piezo'])
            item = self.position_listbox.curselection()[0]
            index = self.position_list[item].index(':')
            cur_pos = self.position_list[item][:index]
            if self.apply_var.get():
                ThreadBG(1)
                time.sleep(0.1)
                self.utils.automatic(self.e518, self.model_lateral, self.model_axial_coarse, self.model_axial_fine)
                if self.fix_var.get():
                    self.replace_z()
                else:
                    self.replace_position()
                self.record_z(cur_pos)
            self.c867.SVO(self.mox, False)
            self.c867.SVO(self.moy, False)
            if self.shutter.get() != '':
                ThreadBG(1, float(self.shutter.get()))
                time.sleep(0.1)
            #self.utils.ph_save(self.filepath.get(), cur_pos)
            self.utils.snap_img(self.filepath.get(), cur_pos)
            self.c867.SVO(self.mox, True)
            self.c867.SVO(self.moy, True)
        self.start_btn['state'] = tk.NORMAL

        interval = float(self.period.get())
        if interval == 0:
            interval = 0.01
        after_id = root.after(int(interval*60*1000), self.start)
        nb_loop = self.nb_loop.get()
        if nb_loop != '':
            if self.nb != int(nb_loop):
                self.nb += 1
            else:
                root.after_cancel(after_id)
                self.nb = 1
                z_dir = os.path.dirname(self.filepath.get())
                z_filename = os.path.join(z_dir, 'z_dict.txt')
                with open(z_filename, 'w') as w:
                    w.write(str(self.z_dict))

    def update_position_list(self, pos_list):
        for i, pos in enumerate(pos_list):
            first = 3
            last = pos.find(":")
            pos = pos[:first+1] + str(i+1) + ":" + pos[last+1:]
            self.position_list[i] = pos
        self.pos_list.set(self.position_list)

    def get_current_position(self):
        pos_dict = {}
        piezo_pos = [self.e518.qPOS(self.x)[self.x],
                     self.e518.qPOS(self.y)[self.y],
                     self.e518.qPOS(self.z)[self.z]]
        motor_pos = [self.c867.qPOS(self.mox)[self.mox],
                     self.c867.qPOS(self.moy)[self.moy]]
        pos_dict['piezo'] = piezo_pos
        pos_dict['motor'] = motor_pos
        return pos_dict

    def get_formatted_position(self, index, motor=None, piezo=None):
        if motor is None and piezo is None:
            pos_dict = self.get_current_position()
            piezo = pos_dict['piezo']
            motor = pos_dict['motor']
        space = 20
        white_space = ' '*(space-len(str(motor).replace('-', '')))
        first = f' Pos{index + 1}: Motor: {motor}{white_space}'
        second = f'/ Piezo: {piezo}'
        pos = first + second
        return pos

    @staticmethod
    def get_valid_position(pos):
        position_dic = {}
        index1 = pos.index('Motor: ') + 7
        index2 = pos.index('/')
        motor_coord = eval(pos[index1:index2])
        piezo_coord = eval(pos[index2+9:])
        position_dic['motor'] = motor_coord
        position_dic['piezo'] = piezo_coord
        return position_dic

    def save(self):
        filename = filedialog.asksaveasfilename(initialdir="/", title="Save file")
        if filename.find('txt') == -1:
            filename = filename + '.txt'
        if filename.split('.')[-1] == 'txt':
            with open(filename, 'w') as w:
                content = self.position_list
                cont = '\n' + str([self.pre_ref, self.cur_ref])
                w.write(str(content)+cont)

    def load(self):
        file_open_dia = filedialog.askopenfilename(initialdir="/", title="Select file")
        if file_open_dia.split('.')[-1] == 'txt':
            self.params_load(file_open_dia)

    def params_load(self, file_open_dia):
        with open(file_open_dia, 'r') as r:
            content = r.read().splitlines()
        self.position_list = eval(content[0])
        self.pos_list.set(self.position_list)
        ref_pos = eval(content[1])
        self.pre_ref = ref_pos[0]
        self.cur_ref = ref_pos[1]
        if self.pre_ref == '':
            self.pre_ref = self.cur_ref
        self.ref_position.set(str(self.cur_ref))

    def move(self, motor, piezo):
        self.c867.VEL(self.mox, 0.05)
        self.c867.VEL(self.moy, 0.05)
        self.c867.MOV(self.mox, motor[0])
        self.c867.MOV(self.moy, motor[1])
        pitools.waitontarget(self.c867)
        self.e518.MOV(self.x, piezo[0])
        self.e518.MOV(self.y, piezo[1])
        self.e518.MOV(self.z, piezo[2])
        pitools.waitontarget(self.e518)

    def apply_machine(self):
        if self.apply_var.get():
            self.apply_var.set(False)
            self.apply.toggle()
            model_lateral_path = r'C:\Users\NBP\Desktop\save_test\model\lateral\model_lateral.hdf5'
            model_axial_coarse_path = r'C:\Users\NBP\Desktop\save_test\model\axial_coarse\model_axial_coarse.hdf5'
            model_axial_fine_path = r'C:\Users\NBP\Desktop\save_test\model\aixal_fine\model_axial_fine.hdf5'
            self.model_lateral = load_model(model_lateral_path)
            self.model_axial_coarse = load_model(model_axial_coarse_path)
            self.model_axial_fine = load_model(model_axial_fine_path)
            print('models successfully loaded')
        else:
            self.apply_var.set(True)
            self.apply.toggle()

    def fix_xy(self):
        if self.fix_var.get():
            self.fix_var.set(False)
            self.fix.toggle()
        else:
            self.fix_var.set(True)
            self.fix.toggle()


class JoyStick(tk.Frame):

    def __init__(self, root, stage):
        self.stage = stage
        self.x = self.stage.axes[1]
        self.y = self.stage.axes[0]
        self.x_range = [self.stage.qTMN(self.x)[self.x], self.stage.qTMX(self.x)[self.x]]
        self.y_range = [self.stage.qTMN(self.y)[self.y], self.stage.qTMX(self.y)[self.y]]
        self.nb_axis = len(self.stage.axes)
        if self.nb_axis == 3:
            self.z = self.stage.axes[2]
        if self.stage.HasHIN():
            self.stage.HIN(self.x, False)
            self.stage.HIN(self.y, False)

        self.popup_win = tk.Toplevel(root)
        dev = self.stage.devname.split('.')[0]
        self.popup_win.wm_title(f"{dev}  Controller")
        self.popup_win.geometry('320x320')
        self.popup_win.configure(bg='white')
        self.popup_win.resizable(0, 0)
        self.popup_win.update()
        self.wn_size = self.popup_win.winfo_width()
        self.wn_pos = [self.popup_win.winfo_rootx(), self.popup_win.winfo_rooty()]
        self.radius = 110
        pad = 200
        self.canvas_range = tk.Canvas(self.popup_win,
                                      bg='white',
                                      borderwidth=0,
                                      highlightthickness=0)
        range_pos = (self.wn_size - (self.radius + pad)) * 0.5
        range_size = (self.radius + pad)
        relsize = range_size / self.wn_size
        self.canvas_range.place(x=range_pos, y=range_pos, relwidth=relsize, relheight=relsize)
        self.create_circle(range_size // 2, range_size // 2, self.radius, self.canvas_range, None)

        if self.nb_axis == 3:
            self.canvas_range.bind('<Enter>', self.bound_to_mousewheel)
            self.canvas_range.bind('<Leave>', self.unbound_to_mousewheel)

        self.dot = tk.Canvas(self.canvas_range,
                             bg='white',
                             borderwidth=0,
                             highlightthickness=0)
        size_ratio = 3
        dot_size = range_size / size_ratio
        self.dot_pos = (range_size - dot_size) * 0.5
        self.dot.place(x=self.dot_pos, y=self.dot_pos, relwidth=1 / size_ratio, relheight=1 / size_ratio)
        self.create_circle(dot_size // 2, dot_size // 2, dot_size * 0.8 // 2, self.dot, 'black')
        self.dot.bind("<Motion>", self.mouse_appearance)
        self.dot.bind("<B1-Motion>", self.drag)
        self.dot.bind("<ButtonRelease-1>", self.centralize)
        self.offset = dot_size - range_pos - dot_size * 0.2 * 2

        self.generator = 0
        self.pressed = False
        self.increment_x = 0
        self.increment_y = 0

    @staticmethod
    def create_circle(x, y, r, canvas, fill):  # center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return canvas.create_oval(x0, y0, x1, y1, outline='black', width=2, fill=fill)

    def bound_to_mousewheel(self, event):
        self.canvas_range.bind("<MouseWheel>", self.set_piezo_axis)

    def unbound_to_mousewheel(self, event):
        self.canvas_range.unbind("<MouseWheel>")
        #pitools.waitontarget(self.stage, axes=[self.z])

    def drag(self, event):
        self.pressed = True
        if self.generator == 0:
            self.set_stage()
        cur_wn_pos = [self.popup_win.winfo_rootx(), self.popup_win.winfo_rooty()]
        if cur_wn_pos != self.wn_pos:
            self.wn_pos = cur_wn_pos
        x = event.widget.winfo_pointerx() - self.wn_pos[0] - self.offset
        y = event.widget.winfo_pointery() - self.wn_pos[1] - self.offset
        x, y = self.get_coord(x, y)
        event.widget.place(x=x, y=y)

    def mouse_appearance(self, event):
        self.dot.config(cursor="hand2")

    def centralize(self, event):
        self.pressed = False
        self.dot.place(x=self.dot_pos, y=self.dot_pos)
        pitools.waitontarget(self.stage, axes=[self.x, self.y])

    def get_coord(self, x, y):
        delta_x = self.dot_pos - x
        delta_y = self.dot_pos - y
        radius = (delta_x ** 2 + delta_y ** 2) ** 0.5
        ratio = radius / self.radius
        if ratio <= 1:
            self.increment_x = (x - self.dot_pos) / self.radius
            self.increment_y = (self.dot_pos - y) / self.radius
            return x, y
        else:
            if delta_x < 0:
                edge_x = abs(delta_x / ratio) + self.dot_pos
            else:
                edge_x = self.dot_pos - (delta_x / ratio)
            if delta_y < 0:
                edge_y = abs(delta_y / ratio) + self.dot_pos
            else:
                edge_y = self.dot_pos - (delta_y / ratio)
            self.increment_x = (edge_x - self.dot_pos) / self.radius
            self.increment_y = (self.dot_pos - edge_y) / self.radius
            return edge_x, edge_y

    def set_stage(self):
        cur_pos = [self.stage.qPOS(self.x)[self.x],
                   self.stage.qPOS(self.y)[self.y]]
        self.generator = 0
        if self.pressed:
            if self.nb_axis == 3:
                increment_x = -self.increment_x
                increment_y = -self.increment_y
                unit_x = unit_y = self.stage.qVEL(self.x)[self.x] * 4
            else:
                increment_x = self.increment_x
                increment_y = self.increment_y
                unit_x = 0.02 * (1 + abs(increment_x))
                unit_y = 0.02 * (1 + abs(increment_y))
                self.stage.VEL(self.x, unit_x * abs(increment_x))
                self.stage.VEL(self.y, unit_y * abs(increment_y))
            target_x = cur_pos[0] + unit_x * increment_x
            target_y = cur_pos[1] + unit_y * increment_y
            if self.x_range[0] < target_x < self.x_range[1]:
                self.stage.MOV(self.x, target_x)
            if self.y_range[0] < target_y < self.y_range[1]:
                self.stage.MOV(self.y, target_y)
            self.generator = root.after(10, self.set_stage)
        else:
            if self.generator != 0:
                root.after_cancel(self.generator)

    def set_piezo_axis(self, event):
        cur_z = self.stage.qPOS(self.z)[self.z]
        target_z = cur_z + event.delta / 1200
        self.stage.MOV(self.z, target_z)


class Main(tk.Frame):

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        popup_win = tk.Toplevel(root)
        popup_win.wm_title("JoyStick Control")
        popup_win.geometry('350x200')
        popup_win.configure(bg='white')
        popup_win.resizable(0, 0)

        sn_label_1 = tk.Label(popup_win, text='C867 Serial Number:',
                              bg='white', font='Arial 12', )
        sn_label_1.place(relx=0.05, rely=0.2)
        self.sn_entry_1 = tk.Entry(popup_win, width=20)
        self.sn_entry_1.insert(0, '0120027194')
        self.sn_entry_1.place(relx=0.5, rely=0.22)

        sn_label_2 = tk.Label(popup_win, text='E518 Serial Number:',
                              bg='white', font='Arial 12', )
        sn_label_2.place(relx=0.05, rely=0.5)
        self.sn_entry_2 = tk.Entry(popup_win, width=20)
        self.sn_entry_2.insert(0, '120027848')
        self.sn_entry_2.place(relx=0.5, rely=0.52)

        enter_btn = tk.Button(popup_win, text='Enter',
                              font='Arial 12', width=12,
                              command=lambda: [self.get_joystick(),
                                               popup_win.destroy()])
        enter_btn.place(relx=0.52, rely=0.78)

        popup_win.protocol("WM_DELETE_WINDOW", lambda: root.destroy())

    def get_joystick(self):
        sn_1 = self.sn_entry_1.get()
        sn_2 = self.sn_entry_2.get()
        c867 = GCSDevice()
        c867.ConnectUSB(serialnum=sn_1)
        print('connected: {}'.format(c867.qIDN().strip()))
        pitools.startup(c867, stages=None, refmodes=None)
        JoyStick(root, c867)

        e518 = GCSDevice()
        e518.ConnectUSB(serialnum=sn_2)
        print('connected: {}'.format(e518.qIDN().strip()))
        pitools.startup(e518, stages=None, refmodes=None)
        JoyStick(root, e518)

        PositionList(root, c867, e518)


if __name__ == '__main__':
    root = tk.Tk()
    root.iconify()
    Main(root)
    root.mainloop()
