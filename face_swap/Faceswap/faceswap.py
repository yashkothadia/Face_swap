import os
import cv2
import copy
import insightface 
import onnxruntime 
import numpy as np
from PIL import Image
from typing import List, Union
from django.http import JsonResponse

def get_face_swap_model(model_path:str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def get_face_analyser(model_path:str, providers, det_size=(320,320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers = providers)
    face_analyser.prepare(ctx_id=0,det_size=det_size)
    return face_analyser


def get_single_face(face_analyser,frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None
    
def get_many_face(face_analyser,frame:np.ndarray):
    
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def flatten_nested_bbox_list(input_list):
    return [item[0] for item in input_list]
    
def swap_face(face_swapper,source_faces,target_faces,source_index,target_index,temp_frame):
    source_face = source_faces[source_index]
    targate_face = target_faces[target_index]
    return face_swapper.get(temp_frame,targate_face,source_face,paste_back=True)

def process(source_img: Union[Image.Image, List],
            target_img: Image.Image,
            source_indexes: str,
            target_indexes: str,
            model:str):
    
    # load machine default aavailable providers
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = get_face_analyser(model, providers)

    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),model)
    face_swapper = get_face_swap_model(model_path)

    # read targate image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # detect faces that will be replaced in the target image
    target_faces = get_many_face(face_analyser,target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img,list) and num_source_images == num_target_faces:
            print("Replacing faces in target image from the left side to the right by order")
            source_faces = [get_many_face(face_analyser,cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR)) for i in range(num_source_images)]
            source_faces = flatten_nested_bbox_list(source_faces)

            if source_faces is None:
                raise Exception("No source faces found")
            
            for i in range(num_target_faces):
                source_index = i
                target_index = i

                temp_frame = swap_face(face_swapper,
                                       source_faces,
                                       target_faces,
                                       source_index,
                                       target_index,
                                       temp_frame)
                
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_face(face_analyser,cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            print(f"Source faces: {num_source_faces}")
            print(f"Target faces: {num_target_faces}")

            if source_faces is None:
                raise Exception("No source faces found")
            
            if target_indexes == "-1":
                if num_source_faces == 1:
                    print("Replacing all faces in target with the same face from the source image")
                    num_iterations = num_target_faces 
                elif num_source_faces < num_target_faces:
                    print("There are less faces in the source image than the target image, replacing as many as we can")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    print("There are less faces in the target image than the source image, replacing as many as we can")
                    num_iterations = num_target_faces
                else:
                    print("Replacing all faces in the target image with the faces from the source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(face_swapper,
                                       source_faces,
                                       target_faces,
                                       source_index,
                                       target_index,
                                       temp_frame)
                    
            else:
                print("Replacing specific face(s) in the target image with specific face(s) from the source image")

                if source_indexes == "-1":
                    source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception("Number of source indexes is greater than the number of faces in the source image")

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception("Number of target indexes is greater than the number of faces in the target image")

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(f"Source index {source_index} is higher than the number of faces in the source image")

                        if target_index > num_target_faces-1:
                            raise ValueError(f"Target index {target_index} is higher than the number of faces in the target image")

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            return JsonResponse({"message": "Unsupported face configuration"}, status=500)
        
        result = temp_frame

    else:
        print("No targgt image found")

    result_image = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    return result_image


