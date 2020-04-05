from tqdm import tqdm


def train(criterion, optimizer, batch_generator, n_epochs, device, scheduler=None):
    losses = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        n_batches = 0
        for (x_idx, x), (y_idx, y) in batch_generator:
            optimizer.zero_grad()
            loss = criterion(x_idx, x.to(device), y_idx, y.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

        if scheduler is not None:
            scheduler.step()

    return losses
