import numpy as np
import torch
import torch.nn as nn


def yolo_forward(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    scale_x_y,
    only_objectness=1,
    validation=False,
):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, h, w]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, h, w]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, h, w]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * h * w]
    det_confs = det_confs.view(batch, num_anchors * h * w)

    # Shape: [batch, num_anchors * num_classes, h, w]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, h * w]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, h * w)
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
        batch, num_anchors * h * w, num_classes
    )
    # Apply sigmoid(), exp() and softmax() to slices
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(
        np.expand_dims(
            np.expand_dims(np.linspace(0, w - 1, w), 0).repeat(h, 0), 0
        ),
        axis=0,
    )
    grid_y = np.expand_dims(
        np.expand_dims(
            np.expand_dims(np.linspace(0, h - 1, h), 1).repeat(w, 1), 0
        ),
        axis=0,
    )
    # grid_x = torch.linspace(0, w - 1, w).reshape(1, 1, 1, w).repeat(
    #     1, 1, h, 1
    # )
    # grid_y = torch.linspace(0, h - 1, h).reshape(1, 1, h, 1).repeat(
    #     1, 1, 1, w
    # )

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, h, w]
        bx = bxy[:, ii : ii + 1] + torch.tensor(
            grid_x, device=device, dtype=torch.float32
        )
        # Shape: [batch, 1, h, w]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(
            grid_y, device=device, dtype=torch.float32
        )
        # Shape: [batch, 1, h, w]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, h, w]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    # Shape: [batch, num_anchors, h, w]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, h, w]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, h, w]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, h, w]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, h, w]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, h, w]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= w
    by_bh /= h

    # Shape: [batch, num_anchors * h * w, 1]
    bx = bx_bw[:, :num_anchors].view(batch, num_anchors * h * w, 1)
    by = by_bh[:, :num_anchors].view(batch, num_anchors * h * w, 1)
    bw = bx_bw[:, num_anchors:].view(batch, num_anchors * h * w, 1)
    bh = by_bh[:, num_anchors:].view(batch, num_anchors * h * w, 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(
        batch, num_anchors * h * w, 1, 4
    )
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * h * w, 1, 4]
    # cls_confs: [batch, num_anchors * h * w, num_classes]
    # det_confs: [batch, num_anchors * h * w]

    det_confs = det_confs.view(batch, num_anchors * h * w, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * h * w, 1, 4]
    # confs: [batch, num_anchors * h * w, num_classes]

    return boxes, confs


class YoloLayer(nn.Module):
    def __init__(
        self,
        anchor_mask=[],
        num_classes=0,
        anchors=[],
        num_anchors=1,
        stride=32,
        model_out=False,
    ):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[
                m * self.anchor_step : (m + 1) * self.anchor_step
            ]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward(
            output,
            self.thresh,
            self.num_classes,
            masked_anchors,
            len(self.anchor_mask),
            scale_x_y=self.scale_x_y,
        )
