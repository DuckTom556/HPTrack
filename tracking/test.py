import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '../')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    #tracker_name：跟踪器名称，对应'/media/ducktom/DATA/SOT/Ubuntu/SeqT/SeqTrack/lib/test/tracker/seqtrack.py'文件
    parser.add_argument('--tracker_name', type=str, default='hptrack',help='Name of tracking method.')
    #tracker_param：跟踪器模型参数名称，对应/media/ducktom/DATA/SOT/Ubuntu/SeqT/SeqTrack/experiments/seqtrack/seqtrack_b256_got.yaml配置文件
    parser.add_argument('--tracker_param', type=str, default='hptrack_b224',help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='uav', help='Name of dataset (otb, nfs, uav, got10k_test, lasot, trackingnet, lasot_extension_subset, tnl2k).')
    #用第sequence个视频序列进行测试，None表示用所有序列进行测试
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')#"Bird1""Box""Coupon""Liquor" None
    #可视化(debug=1)
    parser.add_argument('--debug', type=int, default=0, help='Debug level.') #Doll视频帧非常长-3872
    ##cpu核心数
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)#8

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
