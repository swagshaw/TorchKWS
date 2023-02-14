
import argparse
import torch
import yaml

from torchkws.models import KWSModel
from torchkws.training import Trainer
from torchkws.utils import create_data_loader


def main(args):
    # Load the configuration file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create the data loaders
    train_loader = create_data_loader(config['data']['train'])
    val_loader = create_data_loader(config['data']['val'])
    test_loader = create_data_loader(config['data']['test'])

    # Create the model
    model = KWSModel(
        num_classes=config['model']['num_classes']
    )

    # Move the model to the specified device
    device = torch.device(config['device'])
    model.to(device)

    # Create the optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_step_size'],
        gamma=config['training']['lr_gamma']
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=config['training']['num_epochs'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_dir=config['training']['log_dir']
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    main(args)