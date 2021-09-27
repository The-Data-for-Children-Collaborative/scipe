from fastai.vision.all import *


def model_splitter(model):
    """ Returns list of sets of parameters for ResNet-like models """
    return [params(m) for m in model[0]._modules.values()] + [params(model[1])]


def finetune(df_master, model_master, batch_size, epochs, frozen_epochs, input_shape=(224, 224), seed=42):
    """
    Args:
        df_master (pd.DataFrame): dataframe containing data for all folds to train on
        model_master (torch.nn.Module): model to fine-tune.
        batch_size (int): batch size for training.
        epochs (int): number of epochs to train full model with.
        frozen_epochs (int): number of epochs to initially train regression head for with frozen encoder.
        input_shape ((int,int)): shape of images input to model, assumes ImageNet size.
        seed (int): seed for data split.

    Returns:
        (:obj:`list` of :obj:`torch.nn.Module`): k trained PyTorch models, where k is the number of cross validation folds.

    """
    def ft_subset(df):  # fine-tune the subset of the dataset contained in df
        model = deepcopy(model_master)
        model.cuda()
        model.train()
        dls = ImageDataLoaders.from_df(df,
                                       path='',
                                       fn_col=df.columns.get_loc('file_name'),
                                       label_col=df.columns.get_loc('pop'),
                                       y_block=RegressionBlock,
                                       valid_pct=0.2,
                                       seed=seed,
                                       bs=batch_size,
                                       item_tfms=[Resize(input_shape), DihedralItem(p=1.0)],
                                       batch_tfms=[Normalize.from_stats(*imagenet_stats)],
                                       num_workers=0
                                       )
        learn = Learner(dls, model, loss_func=MSELossFlat(), opt_func=Adam, metrics=[mae], splitter=model_splitter)
        learn.fine_tune(epochs, freeze_epochs=frozen_epochs,
                        cbs=EarlyStoppingCallback(monitor='valid_loss', patience=2))
        model.eval()
        model.cpu()
        return model

    example = torch.Tensor(np.expand_dims(np.zeros((3, input_shape[0], input_shape[1])), 0)).cuda()
    dim = model_master(example).shape[1]  # get length of vector representation
    head = torch.nn.Linear(dim, 1)  # linear regression head for fine-tuning
    model_master = torch.nn.Sequential(model_master, head)
    trained_models = []
    for fold in range(df_master['fold'].max()+1):
        df_train = df_master[(df_master['fold'] != fold) & (df_master['pop'] >= 1) & (df_master['outlier'] == False)]
        trained_models.append(ft_subset(df_train)[0])
    return trained_models
