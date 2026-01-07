from data.data_provider.data_loader import Dataset_SHEERM,Dataset_PSHE,Dataset_CityLearn,Dataset_Estonian
from torch.utils.data import DataLoader

data_dict = {
    'SHEERM': Dataset_SHEERM,
    'PSHE': Dataset_PSHE,
    'CityLearn': Dataset_CityLearn,
    'Estonian': Dataset_Estonian,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        cycle=args.cycle,
    )
    if args.data == 'SHEERM' or args.data == 'PSHE' or args.data == 'CityLearn' or args.data == 'Estonian':
        data_kwargs['pretreatment'] = args.pretreatment
        data_kwargs['cycle'] = args.cycle

    data_set = Data(**data_kwargs)
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
