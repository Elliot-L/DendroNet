import torch
import torch.nn as nn
import torch.nn.functional as F


class RNATracker(nn.Module):
    def __init__(self, conv_dim, input_channel=5, lstm_dim=100, dropout=0.1, decoder_unroll_steps=10, n_classes=1):
        super(RNATracker, self).__init__()

        self.conv_dim = conv_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.decoder_unroll_steps = decoder_unroll_steps

        # First Conv layer
        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=conv_dim, stride=1, kernel_size=10, padding=5)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=3)  # , padding=1)
        self.drop_layer1 = nn.Dropout(p=dropout)

        # Second Conv layer
        self.conv2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, stride=1, kernel_size=10, padding=5)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=3)  # , padding=1)
        self.drop_layer2 = nn.Dropout(p=dropout)

        # bidirectional LSTM, input to this lstm should be [batch_size, length, dim]
        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=lstm_dim, dropout=dropout,
                            num_layers=1, bidirectional=True, batch_first=True)

        # # decoder lstm — unrolls for fixed amount time, g=10, with attention
        # # lstm_dim*2 is the hidden state dimensionality from the bi-directional lstm
        # # lstm_dim*4 is the dimension of inputs to the decoder lstm
        # self.decoder_lstm = nn.LSTMCell(lstm_dim * 4, lstm_dim * 2)
        # self.decoder_unroll_steps = decoder_unroll_steps

        self.drop_layer3 = nn.Dropout(p=dropout)

        # output layer
        self.out = nn.Linear(lstm_dim * 2, n_classes)


    def forward(self, x, batch_len):
        # x input is of shape [batch_size, length, in_channels]
        # mask is of shape [batch_size, length]
        batch_size, max_len, in_channels = x.size()

        # first conv layer -- activation -- max_pooling -- dropout
        x = self.max_pool1(F.relu(self.conv1(x.transpose(1, 2))))
        x = self.drop_layer1(x.transpose(1, 2)).transpose(1, 2)

        # paddings can affect convolution output, use mask to remove this effect
        max_len = (max_len + 1) // 3
        batch_len = (batch_len + 1) // 3
        mask = torch.ones((batch_size, max_len), dtype=torch.int64).cumsum(dim=1).to(x) > batch_len[:, None]
        x = x.masked_fill(mask[:, None, :], 0.)

        # second conv layer -- activation -- max_pooling -- dropout
        x = self.max_pool2(F.relu(self.conv2(x)))
        x = self.drop_layer2(x.transpose(1, 2))

        # same as before: paddings can affect convolution output, use mask to remove this effect
        max_len = (max_len + 1) // 3
        batch_len = (batch_len + 1) // 3
        mask = torch.ones((batch_size, max_len), dtype=torch.int64).cumsum(dim=1).to(x) > batch_len[:, None]
        x = x.masked_fill(mask[:, :, None], 0.)

        # here x is of shape [batch_size, length, out_channels]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, batch_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_x, (hn, cn) = self.lstm(packed_x)
        x = nn.utils.rnn.pad_packed_sequence(packed_x, batch_first=True)[0]
        nb_features = x.size(-1) * 2
        hn = hn.permute(1, 0, 2).reshape(batch_size, -1)
        cn = cn.permute(1, 0, 2).reshape(batch_size, -1)

        # set2set pooling unrolling decoder lstm
        token = torch.zeros(batch_size, nb_features, device=x.device)
        for _ in range(self.decoder_unroll_steps):
            (hn, cn) = self.decoder_lstm(token, (hn, cn))
            # compute attention
            scores = torch.matmul(x, hn[:, :, None])[:, :, 0]
            attention_weights = torch.softmax(scores, dim=-1)
            context_vector = torch.sum(x * attention_weights[:, :, None], dim=1)
            token = torch.cat([context_vector, hn], dim=-1)

        x = self.drop_layer3(token)
        x = self.out(x)
        # binary classification: sigmoid
        # categorical classification: softmax
        # regression: just output a real value
        return x


if __name__ == "__main__":
    model = RNATracker(32, 5)

    # from data_utils import parse_fasta_file, DNATaskDataLoader
    #
    # valid_loader = DNATaskDataLoader(*parse_fasta_file('data/dnase_dataset/Homo_sapiens_valid.fa'),
    #                                  8, 0, shuffle=False)
    # for seq, target, batch_len in valid_loader:
    #     model(seq, batch_len)
    #
    #
    # exit()

    model.eval()
    input_tensor = torch.rand(1, 100, 5)
    batch_len = torch.as_tensor([100])
    output = model(input_tensor, batch_len)
    print(output)

    padded_input = torch.cat([input_tensor, torch.zeros(1, 20, 5)], dim=1)
    output_new = model(padded_input, batch_len)
    print(output_new)
