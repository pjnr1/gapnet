from .datablock import get_datablock, get_subsampling_datablock


def get_dataloaders(input_folder, batch_size=64, valid_pct=0.1, subsample_pct=0.0, input_length=800,
                    num_workers=1, splitter=None, x_block=None, label_func=None):
    if 0.0 <= subsample_pct <= 1.0:
        return get_subsampling_datablock(valid_pct=valid_pct,
                                         subsample_pct=subsample_pct,
                                         desired_length=input_length,
                                         splitter=splitter,
                                         x_block=x_block,
                                         label_func=label_func).dataloaders(input_folder,
                                                                            bs=batch_size,
                                                                            num_workers=num_workers)
    else:
        return get_datablock(valid_pct=valid_pct,
                             splitter=splitter,
                             x_block=x_block,
                             label_func=label_func).dataloaders(input_folder,
                                                                bs=batch_size,
                                                                num_workers=num_workers)
