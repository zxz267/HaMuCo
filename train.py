import torch
from tqdm import tqdm
from config import cfg
from base import Trainer
import collections

def main():
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_loader()
    
    valer = Trainer() 
    valer._make_model(eval=True)
    valer._make_batch_loader(shuffle=False, split='test', drop_last=False)

    for epoch in range(trainer.start_epoch, cfg.total_epoch):
        train_error_results = collections.defaultdict(list)
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(trainer.batch_loader)):
            trainer.optimizer.zero_grad()
            loss, error = trainer.model(inputs, targets)
            for k, v in error.items():
                train_error_results[k].append(v)
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            if not iteration % cfg.print_iter:
                screen = ['[Epoch %d/%d]' % (epoch, cfg.total_epoch),
                          '[Batch %d/%d]' % (iteration, len(trainer.batch_loader)),
                          '[lr %f]' % (trainer.get_lr())]
                screen += ['[%s: %.4f]' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                trainer.logger.info(''.join(screen))

        # print training set error
        for k, v in train_error_results.items():
            errors = torch.stack(v, dim=0).mean()
            trainer.logger.info(f'[Epoch {int(epoch)}]{k}: {errors} px/mm.')
        trainer.schedule.step()

        # validation
        if not epoch % cfg.eval_interval:
            trainer.model.eval()
            val_error_results = collections.defaultdict(list)
            for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(valer.batch_loader)):
                with torch.no_grad():
                    loss, error = trainer.model(inputs, targets)
                for k, v in error.items():
                    val_error_results[k].append(v)

            if epoch in cfg.save_epoch:
                trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch)
            
            # print validation set error
            for k, v in val_error_results.items():
                errors = sum(v) / len(v)
                valer.logger.info(f'[Epoch {int(epoch)}]{k}: {errors} px/mm.')
        
            # continue training
            trainer.model.train()

if __name__ == '__main__':
    main()



