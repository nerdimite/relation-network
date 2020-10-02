import cv2
import os
import numpy as np
import random
import pickle
import argparse

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

# Dataset Parameters
img_size = 75
size = 5
question_size = 10 # 6 for one-hot vector of color, 1 for question type, 3 for question subtype
q_type_idx = 6
sub_q_type_idx = 7
nb_questions = 10
# Possibles Answers : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]

colors = [
    (0,0,255), # red
    (0,255,0), # green
    (255,0,0), # blue
    (0,156,255), # orange
    (128,128,128), # gray
    (0,255,255) # yellow
]

def center_generate(objects):
    '''Generate centers of objects'''
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_sample():
    '''Returns an image and the corresponding questions'''
    
    # Create objects
    objects = []
    img = np.ones((img_size,img_size,3)) * 255
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id,center,'c'))


    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    
    # Non-Relational Questions
    for _ in range(nb_questions):
        
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 0
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        
        if subtype == 0:
            # query shape -> rectangle/circle
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            # query is left side (horizontal position) -> yes/no
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            # query is up side (vertical position) -> yes/no
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    # Relational Questions
    for _ in range(nb_questions):
        
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        rel_questions.append(question)

        if subtype == 0:
            # closest to -> rectangle/circle
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            # furthest from -> rectangle/circle
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            # count -> 1~6
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count += 1 
            answer = count + 4

        rel_answers.append(answer)

    relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)
    img = img / 255.
    sample = (img, relations, norelations)
    
    return sample

def translate_sample(sample, show_img=False):
    '''Translate question/answer vector to english'''
    img, (rel_questions, rel_answers), (norel_questions, norel_answers) = sample
    colors = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
    questions = rel_questions + norel_questions
    answers = rel_answers + norel_answers

    for i, (question, answer) in enumerate(zip(questions, answers)):
        query = f'Q{i}. '
        color = colors[question.tolist()[0:6].index(1)]
        
        # Non-relational questions
        if question[q_type_idx] == 0:
            if question[sub_q_type_idx] == 1:
                query += f'What is the shape of the {color} object?'
            elif question[sub_q_type_idx+1] == 1:
                query += f'Is there a {color} object on the left?'
            elif question[sub_q_type_idx+2] == 1:
                query += f'Is there a {color} object on the top?'
            
                
        # Relational questions
        elif question[q_type_idx] == 1:
            if question[sub_q_type_idx] == 1:
                query += f'What is the closest shape to the {color} object?'
            elif question[sub_q_type_idx+1] == 1:
                query += f'What is the furthest shape from the {color} object?'
            elif question[sub_q_type_idx+2] == 1:
                query += f'How many objects of the same shape as the {color} object are there?'
        
        ans = answer_sheet[answer]
        print(query,'==>', ans)
    
    if show_img:
        cv2.imshow('img', cv2.resize(img,(512,512)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Sort-of-CLEVR Dataset Generator')
    parser.add_argument('--n_train', type=int, default=8000, help='number of train images to generates')
    parser.add_argument('--n_test', type=int, default=1000, help='number of test images to generates')
    args = parser.parse_args()
    
    print('Building Train Dataset...')
    train_data = [build_sample() for _ in range(args.n_train)]
    print('Building Test Dataset...')
    test_data = [build_sample() for _ in range(args.n_test)]
    
    data_dir = './data'
    try:
        os.makedirs(data_dir)
    except:
        print('Directory {} already exists'.format(data_dir))
        
    print('Saving Datasets...')
    filename = os.path.join(data_dir,'sort-of-clevr.pickle')
    with open(filename, 'wb') as f:
        pickle.dump((train_data, test_data), f)
    print('Datasets saved at {}'.format(data_dir))