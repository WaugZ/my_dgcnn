import os
import argparse
import distillation_models as model
import data
import tensorflow as tf
import numpy as np
import random

seed = int(os.getenv("SEED", 12))
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


def check_and_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default="logs/log")
    parser.add_argument('--gpu', type=int, default=0, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--load_teacher_from_checkpoint', type=str, default="false")
    parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)

    # Model Parameters
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
    parser.add_argument('--num_epoch', type=int, default=251, help='Epoch to run [default: 250]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--quantize_delay_t', type=int, default=None, help='Quantization decay, >0 for open [default:0]')
    parser.add_argument('--quantize_delay_s', type=int, default=None, help='Quantization decay, >0 for open [default:0]')
    parser.add_argument('--checkpoint', default=None, help='Restore checkpoint')
    parser.add_argument('--dynamic_t', type=int, default=-1,
                        help="Whether dynamically compute the distance[<0 for yes else for no]")
    parser.add_argument('--dynamic_s', type=int, default=-1,
                        help="Whether dynamically compute the distance[<0 for yes else for no]")
    parser.add_argument('--stn_t', type=int, default=-1,
                        help="whether use STN[<0 for yes else for no]")
    parser.add_argument('--stn_s', type=int, default=-1,
                        help="whether use STN[<0 for yes else for no]")
    parser.add_argument('--scale_t', type=float, default=1., help="dgcnn depth scale")
    parser.add_argument('--scale_s', type=float, default=.5, help="dgcnn depth scale")
    parser.add_argument('--concat_t', type=int, default=1, help="whether concat neighbor's feature 1 for yes else for no")
    parser.add_argument('--concat_s', type=int, default=1, help="whether concat neighbor's feature 1 for yes else for no")

    return parser


def setup(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

    args.load_teacher_from_checkpoint = convert_str_to_bool(args.load_teacher_from_checkpoint)

    check_and_makedir(args.log_dir)
    # check_and_makedir(os.path.dirname(args.checkpoint_path))


def main():
    parser = get_parser()
    args = parser.parse_args()
    setup(args)
    print(args)

    tf.reset_default_graph()
    if args.model_type == "student":
        teacher_model = None
        if args.load_teacher_from_checkpoint:
            teacher_model = model.BigModel(args, "teacher")
            teacher_model.start_session()
            teacher_model.load_model_from_file(args.load_teacher_checkpoint_dir)
            print("Verify Teacher State before Training Student")
            teacher_model.run_inference(teacher_model.sess)
        student_model = model.SmallModel(args, "student")
        student_model.start_session()
        student_model.train(teacher_model)

        if args.load_teacher_from_checkpoint:
            print("Verify Teacher State After Training student Model")
            # teacher_model.run_inference(dataset)
            teacher_model.close_session()
        student_model.close_session()
    else:
        teacher_model = model.BigModel(args, "teacher")
        teacher_model.start_session()
        teacher_model.train()


if __name__ == '__main__':
    main()
    # INVOCATION

    # Teacher
    # python main.py --model_type teacher --checkpoint_dir teachercpt --num_steps 50

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --num_steps 50 --gpu 0

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --load_teacher_from_checkpoint true
    # --load_teacher_checkpoint_dir teachercpt --num_steps 50