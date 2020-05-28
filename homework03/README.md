## score 20.61
* Был развернут TRG
* Увеличено количество эпох до 15


## score 25.86
* Использовал GRU вместо LSTM
* Реализовал простейший attention (dot product between decoder state & encoder output)


## score 26.18
* Использовал GRU вместо LSTM
* Реализовал weighted attention (dot product between decoder state & W_matrinx & encoder output)

## score 28.30
* Использовал GRU вместо LSTM
* general attention from torchnlp

## score 30.10
* Использовал LSTM
* general attention from torchnlp

## score 24
* Попробовал добавить PE, после одной из лекций, понял, что это было бессмысленно

## score 30.00
* Использовал BiLSTM
* general attention from torchnlp
* Output энкодреа просто плюсовал, чтобы перейти от 2*hid_dim к hid_dim
* Hidden энкодера приобразовал к нужному размеру с помощью линейного слоя