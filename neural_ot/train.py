

def train(criterion, optimizer, batch_generator, n_epochs):
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, y in batch_generator:
            optimizer.zero_grad()
            loss = criterion(x, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss)

    return losses
