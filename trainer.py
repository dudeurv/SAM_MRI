{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMB34QkTz0h8SIbgOKpSdF2",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/dudeurv/SAM_MRI/blob/main/trainer.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oFQGscZVGHyq"
      },
      "outputs": [],
      "source": [
        "import argparse\n",
        "import logging\n",
        "import os\n",
        "import random\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.backends.cudnn as cudnn\n",
        "\n",
        "from importlib import import_module\n",
        "\n",
        "from sam_lora_image_encoder import LoRA_Sam\n",
        "from segment_anything import sam_model_registry\n",
        "\n",
        "from trainer import trainer_synapse\n",
        "from icecream import ic\n",
        "\n",
        "parser = argparse.ArgumentParser()\n",
        "parser.add_argument('--root_path', type=str,\n",
        "                    default='/home/sist/kdzhang/data/train_npz_new_224/', help='root dir for data')\n",
        "parser.add_argument('--output', type=str, default='/home/sist/kdzhang/results/SAMed_accelerate')\n",
        "parser.add_argument('--dataset', type=str,\n",
        "                    default='Synapse', help='experiment_name')\n",
        "parser.add_argument('--list_dir', type=str,\n",
        "                    default='./lists/lists_Synapse', help='list dir')\n",
        "parser.add_argument('--num_classes', type=int,\n",
        "                    default=8, help='output channel of network')\n",
        "parser.add_argument('--max_iterations', type=int,\n",
        "                    default=30000, help='maximum epoch number to train')\n",
        "parser.add_argument('--max_epochs', type=int,\n",
        "                    default=200, help='maximum epoch number to train')\n",
        "parser.add_argument('--stop_epoch', type=int,\n",
        "                    default=160, help='maximum epoch number to train')\n",
        "parser.add_argument('--batch_size', type=int,\n",
        "                    default=24, help='batch_size per gpu')\n",
        "parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')\n",
        "parser.add_argument('--deterministic', type=int, default=1,\n",
        "                    help='whether use deterministic training')\n",
        "parser.add_argument('--base_lr', type=float, default=0.005,\n",
        "                    help='segmentation network learning rate')\n",
        "parser.add_argument('--img_size', type=int,\n",
        "                    default=512, help='input patch size of network input')\n",
        "parser.add_argument('--seed', type=int,\n",
        "                    default=1234, help='random seed')\n",
        "parser.add_argument('--vit_name', type=str,\n",
        "                    default='vit_h', help='select one vit model')\n",
        "parser.add_argument('--ckpt', type=str, default='/home/sist/kdzhang/data/sam_vit_h_4b8939.pth',\n",
        "                    help='Pretrained checkpoint')\n",
        "parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')\n",
        "parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')\n",
        "parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')\n",
        "parser.add_argument('--warmup_period', type=int, default=250,\n",
        "                    help='Warp up iterations, only valid whrn warmup is activated')\n",
        "parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')\n",
        "parser.add_argument('--module', type=str, default='sam_lora_image_encoder')\n",
        "parser.add_argument('--dice_param', type=float, default=0.8)\n",
        "\n",
        "parser.add_argument('--lr_exp', type=float, default=0.9, help='The learning rate decay expotential')\n",
        "\n",
        "# acceleration choices\n",
        "parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')\n",
        "parser.add_argument('--compile', action='store_true', help='If activated, compile the training model for acceleration')\n",
        "parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')\n",
        "\n",
        "args = parser.parse_args()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == \"__main__\":\n",
        "    random.seed(args.seed)\n",
        "    np.random.seed(args.seed)\n",
        "    torch.manual_seed(args.seed)\n",
        "\n",
        "    # Create snapshot path\n",
        "    snapshot_path = os.path.join(args.output, f\"exp_{args.vit_name}_bs{args.batch_size}_lr{args.base_lr}_epochs{args.max_epochs}\")\n",
        "    if not os.path.exists(snapshot_path):\n",
        "        os.makedirs(snapshot_path)\n",
        "\n",
        "    # Load the model\n",
        "    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,\n",
        "                                                                num_classes=args.num_classes,\n",
        "                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],\n",
        "                                                                pixel_std=[1, 1, 1])\n",
        "\n",
        "    pkg = import_module(args.module)\n",
        "    net = pkg.LoRA_Sam(sam, args.rank)  # Removed .cuda()\n",
        "    if args.lora_ckpt is not None:\n",
        "        net.load_lora_parameters(args.lora_ckpt)\n",
        "\n",
        "    multimask_output = args.num_classes > 1\n",
        "    low_res = img_embedding_size * 4\n",
        "\n",
        "    # Write configuration to a file\n",
        "    config_file = os.path.join(snapshot_path, 'config.txt')\n",
        "    config_items = [f'{key}: {value}\\n' for key, value in args.__dict__.items()]\n",
        "    with open(config_file, 'w') as f:\n",
        "        f.writelines(config_items)\n",
        "\n",
        "    # Start training\n",
        "    trainer_synapse(args, net, snapshot_path, multimask_output, low_res)"
      ],
      "metadata": {
        "id": "3xsRnN4XGVjU"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}