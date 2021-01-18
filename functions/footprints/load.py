import tensorflow as tf

SEED = 42

def parse_image(img_path): # load image from img_path
    image = tf.io.read_file(img_path)
    image = tf.image.decode_tiff(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return image

def parse_label(img_path): # load label associated with image at img_path
    label_path = tf.strings.regex_replace(img_path, "images", "labels")
    label = tf.io.read_file(label_path)
    label = tf.image.decode_tiff(label, channels=1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    return label

def load_dataset(train_dir, val_dir, target_size, ratio):
    train_dataset = tf.data.Dataset.list_files(train_dir + 'data/' + "*.tif", seed=SEED)
    X_train = train_dataset.map(parse_image)
    Y_train = train_dataset.map(parse_label)

    val_dataset = tf.data.Dataset.list_files(val_dir + 'data/' + "*.tif", seed=SEED)
    X_val = val_dataset.map(parse_image)
    Y_val = val_dataset.map(parse_label)
    
    image_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        target_size=target_size,
        rescale=1./255)
    label_datagen = ImageDataGenerator(
        target_size=targetsize)
    
    return