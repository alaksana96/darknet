import numpy
import cv2
from multiprocessing import Process, Manager
import yaml
from tqdm import tqdm

intPERSON = 0
intHEAD   = 1

manager = Manager()
return_dict = manager.dict()

def generate_annotations(ii, return_dict):
            
    line = return_dict[ii]
    del return_dict[ii]
    
    dictLine = yaml.load(line)

    strID = dictLine['ID']

    img = cv2.imread('crowdhuman_val/{}.jpg'.format(strID),1)

    imgWidth  = img.shape[1]
    imgHeight = img.shape[0]

    # Create .txt label file
    with open('crowdhuman_val/{}.txt'.format(strID), 'w+') as txtf:

        for label in dictLine['gtboxes']:

            if 'extra' in label and 'ignore' in label['extra'] and label['extra']['ignore'] == 1:
                continue
            if 'extra' in label and 'unsure' in label['extra'] and label['extra']['unsure'] == 1:
                continue 

            # Person BB
            px = float(label['fbox'][0])
            py = float(label['fbox'][1])
            pw = float(label['fbox'][2])
            ph = float(label['fbox'][3])

            # Head BB
            hx = float(label['hbox'][0])
            hy = float(label['hbox'][1])
            hw = float(label['hbox'][2])
            hh = float(label['hbox'][3])

            # Absolute person BB
            cpx = px + pw/2
            cpy = py + ph/2

            abspx = cpx / imgWidth
            abspy = cpy / imgHeight
            abspw = pw / imgWidth
            absph = ph / imgHeight  

            abspx = 1 if abspx > 1 else abspx
            abspy = 1 if abspy > 1 else abspy
            abspw = 1 if abspw > 1 else abspw
            absph = 1 if absph > 1 else absph
            abspx = 0.000001 if abspx < 0 else abspx
            abspy = 0.000001 if abspy < 0 else abspy
            abspw = 0.000001 if abspw < 0 else abspw
            absph = 0.000001 if absph < 0 else absph

            # Absolute head BB
            chx = hx + hw/2
            chy = hy + hh/2

            abshx = chx / imgWidth
            abshy = chy / imgHeight
            abshw = hw / imgWidth
            abshh = hh / imgHeight  

            abshx = 1 if abshx > 1 else abshx
            abshy = 1 if abshy > 1 else abshy
            abshw = 1 if abshw > 1 else abshw
            abshh = 1 if abshh > 1 else abshh
            abshx = 0.000001 if abshx < 0 else abshx
            abshy = 0.000001 if abshy < 0 else abshy
            abshw = 0.000001 if abshw < 0 else abshw
            abshh = 0.000001 if abshh < 0 else abshh

            # Write to file
            txtf.write('{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(intPERSON,
                                                                 abspx,
                                                                 abspy,
                                                                 abspw,
                                                                 absph))

            txtf.write('{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(intHEAD,
                                                                 abshx,
                                                                 abshy,
                                                                 abshw,
                                                                 abshh))
        

with open('annotation_val.odgt') as f:
        
    processes = []
    max_iter = 500

    for ii, line in tqdm(enumerate(f)): 

        return_dict[ii] = line
        pcs = Process(target = generate_annotations, args = (ii, return_dict))
        processes.append(pcs)
        pcs.start()
        
        if ii % max_iter == 0:
            
            for jj in range(len(processes)):    
                processes[jj].join()  

            processes = []
            
    for jj in range(len(processes)):    
        processes[jj].join()      
        
    processes = []    