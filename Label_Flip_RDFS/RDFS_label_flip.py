from keras.models import load_model, Model
from glob import glob
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from pathlib import Path


def get_attack_files(result_dir, attack_name):
    os.chdir(result_dir + attack_name + '/npy')
    files_name = [name for name in os.listdir('.') if os.path.isfile(name)]
    files_name.sort()
    images = []
    image_names = []
    for file in files_name:
        if not file == "tunning.csv":
            data = np.load(result_dir + dir + '/npy/' + file)
            array_sum = np.sum(data)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                # os.remove(result_dir + dir + '/npy/'+ file)
                # print(f"Delete index {file}");
                continue
            else:
                images.append(data)
                image_names.append(file)
    return images


if __name__ == '__main__':
    for perc in [10]:
        idx1 = [5, 10, 30, 50, 100, 192]
        sizeII = 50  # different cases (example 20 different cases)
        num_flatten_features = 192
        num_training = 15000
        num_val = 1500
        num_test = 3000

        StammNet_model_path = f'models/trained_label_flipping_{perc}_percent.h5'
        StammNet_model = load_model(StammNet_model_path)
        Model.summary(StammNet_model)
        StammNet_model_fe = Model(inputs=StammNet_model.input, outputs=StammNet_model.get_layer(
            'flatten').output)  # takes bounds[0,1] shape(1,128,128,1) as input
        Model.summary(StammNet_model_fe)
        temp_path = Path(f"./RDFS_Flipping/")
        dataset_path = Path(f"./Dataset/{perc}_Percent")
        RDFS_path = Path(f"./RDFS_Flipping/{perc}_Percent/")
        Path.mkdir(temp_path.joinpath(f"{perc}_Percent"), exist_ok=True, parents=True)

        size = 32
        save_features = True
        load_features = True
        if save_features:
            flipped_indices_train = np.load(dataset_path.joinpath("train", "flipped_pristine_data_indices.npy"))
            flipped_indices_val = np.load(dataset_path.joinpath("val", "flipped_pristine_data_indices.npy"))

            ''' image lists '''
            StammNet_imlist_pristine_Tr = np.load(dataset_path.joinpath("train", "pristine", "data.npy"))
            StammNet_imlist_pristine_Val = np.load(dataset_path.joinpath("val", "pristine", "data.npy"))
            StammNet_imlist_pristine_Te = np.load(dataset_path.joinpath("test_pure", "pristine", "data.npy"))

            StammNet_imlist_resize08_Tr = np.load(dataset_path.joinpath("train", "fake", "data.npy"))
            StammNet_imlist_resize08_Val = np.load(dataset_path.joinpath("val", "fake", "data.npy"))
            StammNet_imlist_resize08_Te = np.load(dataset_path.joinpath("test_pure", "fake", "data.npy"))

            StammNet_imlist_Flipped = np.load(dataset_path.joinpath("test", "pristine", "data.npy"))

            ###########################################################
            # Pristine TR + Val + Test
            ###########################################################

            feStamm_pristine_Tr = np.zeros(num_flatten_features)
            for idx, im_file in enumerate(StammNet_imlist_pristine_Tr[:num_training]):
                if idx not in flipped_indices_train:
                    print('dealing with number {} image in StammNet_imlist_pristine_Tr'.format(idx))
                    feStamm_pristine_Tr = np.vstack(
                        (feStamm_pristine_Tr, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
            feStamm_pristine_Tr = np.delete(feStamm_pristine_Tr, 0, 0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_pristine_Tr.npy'), feStamm_pristine_Tr)
            print(f"Num Training Data: {len(feStamm_pristine_Tr)}")
            num_training = len(feStamm_pristine_Tr)

            feStamm_pristine_Val = np.zeros(num_flatten_features)
            for idx, im_file in enumerate(StammNet_imlist_pristine_Val[:num_val]):
                if idx not in flipped_indices_val:
                    print('dealing with number {} image in StammNet_imlist_pristine_Val'.format(idx))
                    feStamm_pristine_Val = np.vstack(
                        (feStamm_pristine_Val, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
            feStamm_pristine_Val = np.delete(feStamm_pristine_Val, 0, 0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_pristine_Val.npy'), feStamm_pristine_Val)
            print(f"Num Training Data: {len(feStamm_pristine_Val)}")
            num_val = len(feStamm_pristine_Val)

            feStamm_pristine_Te = np.zeros(num_flatten_features)
            for idx, im_file in enumerate(StammNet_imlist_pristine_Te[:num_test]):
                print('dealing with number {} image in StammNet_imlist_pristine_Te'.format(idx))
                feStamm_pristine_Te = np.vstack(
                    (feStamm_pristine_Te, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
            feStamm_pristine_Te = np.delete(feStamm_pristine_Te, 0, 0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_pristine_Te.npy'), feStamm_pristine_Te)

            concatenate_Pristine = np.concatenate(
                (feStamm_pristine_Tr[0:num_training, :], feStamm_pristine_Val[0:num_val, :],
                 feStamm_pristine_Te[0:num_test, :]), axis=0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_pristine_Concat.npy'), concatenate_Pristine)

            #########################################################################
            #      Resize08 flatten features  (Tr+Val+Te)
            #########################################################################
            feStamm_resize08_Tr = np.zeros(num_flatten_features)
            for idx, im_file in enumerate(StammNet_imlist_resize08_Tr[:num_training]):
                print('dealing with number {} image in StammNet_imlist_resize08_Tr'.format(idx))
                feStamm_resize08_Tr = np.vstack(
                    (feStamm_resize08_Tr, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
            feStamm_resize08_Tr = np.delete(feStamm_resize08_Tr, 0, 0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_resize08_Tr.npy'), feStamm_resize08_Tr)

            feStamm_resize08_Val = np.zeros(num_flatten_features)
            for idx, im_file in enumerate(StammNet_imlist_resize08_Val[:num_val]):
                print('dealing with number {} image in StammNet_imlist_resize08_Val'.format(idx))
                feStamm_resize08_Val = np.vstack(
                    (feStamm_resize08_Val, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
            feStamm_resize08_Val = np.delete(feStamm_resize08_Val, 0, 0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_resize08_Val.npy'), feStamm_resize08_Val)

            feStamm_resize08_Te = np.zeros(num_flatten_features)
            for idx, im_file in enumerate(StammNet_imlist_resize08_Te[:num_test]):
                print('dealing with number {} image in StammNet_imlist_resize08_Te'.format(idx))
                feStamm_resize08_Te = np.vstack((feStamm_resize08_Te, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
            feStamm_resize08_Te = np.delete(feStamm_resize08_Te, 0, 0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_resize08_Te.npy'), feStamm_resize08_Te)

            concatenate_resize08 = np.concatenate(
                (feStamm_resize08_Tr[0:num_training, :], feStamm_resize08_Val[0:num_val, :], feStamm_resize08_Te[0:num_test, :]), axis=0)
            np.save(RDFS_path.joinpath('StammNet_model_fe_resize08_Concat.npy'), concatenate_resize08)

        feStamm_Flipped = np.zeros(num_flatten_features)
        flip_idx = np.random.choice(StammNet_imlist_Flipped.shape[0], 500)
        StammNet_imlist_Flipped = StammNet_imlist_Flipped[flip_idx]
        for idx, im_file in enumerate(StammNet_imlist_resize08_Te[:num_test]):
            print('dealing with number {} image in Flipped'.format(idx))
            feStamm_Flipped = np.vstack(
                (feStamm_Flipped, StammNet_model_fe.predict(im_file.reshape(1, size, size, 1))))
        feStamm_Flipped = np.delete(feStamm_Flipped, 0, 0)
        np.save(RDFS_path.joinpath('StammNet_model_fe_Flipped.npy'), feStamm_Flipped)

        if load_features:
            ########################################################################
            #     Load all features
            #############################################################

            pristine = np.load(RDFS_path.joinpath('StammNet_model_fe_pristine_Concat.npy'))
            resize08 = np.load(RDFS_path.joinpath('StammNet_model_fe_resize08_Concat.npy'))
            Flipped = np.load(RDFS_path.joinpath('StammNet_model_fe_Flipped.npy'))

            print('pristine shape', pristine.shape)
            print('resize08 shape', resize08.shape)
            print('JSMA01 shape', Flipped.shape)

            assert len(pristine[0]) == pristine.shape[1]
            index = np.array(range(pristine.shape[1]))
            for I in idx1:
                Feature_Path = RDFS_path.joinpath(str(I), f"inter_feature_{sizeII}")
                Path.mkdir(Feature_Path, parents=True, exist_ok=True)
                print(f"Starting pattern of size {I}")
                for idx in np.arange(sizeII):
                    Stamm_random_pattern = np.random.choice(index, size=I, replace=False)
                    Stamm_random_pattern.sort()
                    np.save(Feature_Path.joinpath(f'StammNet_random_pattern_{idx}.npy'), Stamm_random_pattern)

                    '''Using pattern generate subset'''
                    Stamm_random_pattern = np.load(Feature_Path.joinpath(f'StammNet_random_pattern_{idx}.npy'))

                    Stamm_pristine = np.array([pristine[:, j] for j in Stamm_random_pattern]).T
                    Stamm_resize08 = np.array([resize08[:, j] for j in Stamm_random_pattern]).T
                    Stamm_Flipped = np.array([Flipped[:, j] for j in Stamm_random_pattern]).T

                    np.save(Feature_Path.joinpath(f'StammNet_subset_pristine_{idx}.npy'), Stamm_pristine)
                    np.save(Feature_Path.joinpath(f'StammNet_subset_resize08_{idx}.npy'), Stamm_resize08)
                    np.save(Feature_Path.joinpath(f'StammNet_subset_flipped_{idx}.npy'), Stamm_Flipped)

    print(f"Num Training: {num_training}")
    print(f"Num Val: {num_val}")
    print(f"Num Test: {num_test}")