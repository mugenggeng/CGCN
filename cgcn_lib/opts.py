


def parse_opt(parser):
    # parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.004)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=10)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=2)
    # output settings
    parser.add_argument('--eval', type=str, default='validation')
    parser.add_argument('--output', type=str, default="../gtadData/activity_output")
    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="../gtadData/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="../gtadData/activitynet_annotations/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="../gtadData/csv_mean_100.hdf5")

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=400) 
    parser.add_argument(
        '--h_dim_1d',
        type=int,
        default=256) 
    parser.add_argument(
        '--h_dim_2d',
        type=int,
        default=128) 
    parser.add_argument(
        '--h_dim_3d',
        type=int,
        default=512) 

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.8)
    parser.add_argument(
        '--result_file',
        type=str,
        default="result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="evaluation_result.pdf")

    # anchors
    parser.add_argument('--max_duration', type=int, default=100)  # anet: 100 snippets
    parser.add_argument('--min_duration', type=int, default=0)  # anet: 100 snippets


    # ablation settings

    parser.add_argument(
        '--goi_samp',
        type=int,
        default=0) # 0: sample all frame; 1: sample each output position
    parser.add_argument(
        '--goi_style',
        type=int,
        default=1)  # 0: no context, 1: last layer context, 2: all layer context
    parser.add_argument(
        '--kern_2d',
        type=int,
        default=1) # 3
    parser.add_argument(
        '--pad_2d',
        type=int,
        default=0) # 1


    # few shot setting

    parser.add_argument(
        '--shot',
        type=int,
        default=1) # 2

    parser.add_argument(
        '--episode',
        type=int,
        default=100) # 50

    parser.add_argument(
        '--meta_learn',
        type=bool,
        default=False) # 50

    parser.add_argument(
        '--multi_instance',
        type=bool,
        default=False) # common Instance or common Multi-Instance

    parser.add_argument(
        '--use_trans',
        type=bool,
        default=True) # use self-attention or not

    parser.add_argument(
        '--cross_domain',
        type=bool,
        default=False) # use cross-domain or not

    parser.add_argument(
        '--is_trimmed',
        type=bool,
        default=False) # use trimmed support or untrimmed if False

    args = parser.parse_args()

    return args

