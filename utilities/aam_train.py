import menpo.io as mio
from menpofit.aam import HolisticAAM
from menpo.feature import fast_dsift

def process(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    #labeller(image, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
    return image

path_to_images = '../../../data/lfpw/trainset_34/'
training_images = mio.import_images(path_to_images, verbose=True)
training_images = training_images.map(process)

aam = HolisticAAM(training_images,holistic_features=fast_dsift, verbose=True,scales=1,
                  max_shape_components=16, max_appearance_components=104)
mio.export_pickle(aam, "aam.pkl", overwrite=True, protocol=4)
