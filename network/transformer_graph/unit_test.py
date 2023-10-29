import torch
import unittest
from main import *


def cross_entropy_loss( pred, trg):
    """
    Compute the binary cross-entropy loss between predictions and targets.

    Args:
    - pred (torch.Tensor): Model predictions.
    - trg (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: Computed BCE loss.
    """
    weights = (trg[:, 1:].clone() * 1) + 1
    loss = F.binary_cross_entropy(torch.sigmoid(pred[:, :-1]), get_clipped_adj_matrix(trg[:, 1:]), reduction='none',
                                  weight=weights)

    # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
    pad_mask = get_pad_mask(trg[:, 1:, 0], pad_idx=self.pad_idx)
    sub_mask = get_subsequent_mask(trg[:, :, 0])[:, 1:, :]
    mask = pad_mask.unsqueeze(-1).expand(-1, -1, loss.shape[2]) & sub_mask

    # mask 적용
    masked_loss = loss * mask.float()
    # 손실의 평균 반환
    return masked_loss.sum() / mask.float().sum()

class TestModelLossAndHelpers(unittest.TestCase):

    def setUp(self):
        # Sample data can be initialized here
        self.pred = torch.rand((32, 10))  # Sample prediction tensor
        self.trg = torch.randint(0, 2, (32, 10)).float()  # Sample target tensor

    def test_get_pad_mask(self):
        # Sample input and expected output
        input_tensor = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        expected_output = torch.tensor([[False, False, True], [False, False, False], [False, True, True]])

        output = get_pad_mask(input_tensor)
        self.assertTrue(torch.equal(output, expected_output))

    def test_get_subsequent_mask(self):
        # Sample input and expected output for a sequence length of 3
        input_tensor = torch.tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
        expected_output = torch.tensor([
            [True, False, False],
            [True, True, False],
            [True, True, True]
        ])

        output = get_subsequent_mask(input_tensor)
        print(f"input: {input_tensor}\n expected_output: {expected_output}\n output: {output}")
        self.assertTrue(torch.equal(output, expected_output))

    def test_cross_entropy_loss(self):
        # Mock the helper functions to isolate the cross_entropy_loss testing
        def mock_get_pad_mask(tensor, pad_idx=0):
            return tensor == pad_idx

        def mock_get_subsequent_mask(tensor):
            size = tensor.size(1)
            mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
            return mask

        original_get_pad_mask = globals().get('get_pad_mask')
        original_get_subsequent_mask = globals().get('get_subsequent_mask')
        globals()['get_pad_mask'] = mock_get_pad_mask
        globals()['get_subsequent_mask'] = mock_get_subsequent_mask

        # Call the cross_entropy_loss function and check if the loss value is within expected range (this is a basic test)
        loss = Trainer.cross_entropy_loss(self.pred, self.trg)
        self.assertTrue(0 <= loss.item() <= 1)

        # Restore the original functions
        globals()['get_pad_mask'] = original_get_pad_mask
        globals()['get_subsequent_mask'] = original_get_subsequent_mask


if __name__ == "__main__":
    unittest.main()