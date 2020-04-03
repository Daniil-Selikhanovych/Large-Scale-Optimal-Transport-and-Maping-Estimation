

def train(criterion, optimizer, batch_generator, batch_size, n_epochs, n_batches_per_epoch):
    losses = []
    for epoch in range(n_epochs):
        epoch_avg_loss = 0
        for x, y in batch_generator(n_batches_per_epoch, batch_size):
            optimizer.zero_grad()
            loss = criterion(x, y)
            loss.backward()
            optimizer.step()

            epoch_avg_loss += loss.item()

        epoch_avg_loss /= n_batches_per_epoch
        losses.append(epoch_avg_loss)

    return losses
