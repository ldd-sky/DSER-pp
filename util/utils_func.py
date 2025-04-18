import numpy as np
import torch
import random
import scipy.io as scio
import torch.nn.functional as F
from util.event import EventSequence


def load_bin(raw_file):
    f = open(raw_file, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()

    raw_data = np.uint32(raw_data)
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    all_ts = all_ts / 1e6  # µs -> s
    all_p = all_p.astype(np.float64)
    all_p[all_p == 0] = -1
    events = np.column_stack((all_x, all_y, all_ts, all_p))
    return events


def binary_search_numpy_array(t, l, r, x, side='left'):
    """
    Binary search sorted numpy array
    """
    if r is None:
        r = len(t) - 1
    while l <= r:
        mid = l + (r - l) // 2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def index_patch(patch, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # print(points.shape, idx.shape)
    device = patch.device
    B = patch.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # print(B, view_shape, repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # print(batch_indices.shape, idx.shape, patch.shape)
    new_patch = patch[batch_indices, idx, :]
    return new_patch


def matDataToArray(filePath):
    data = scio.loadmat(filePath, verify_compressed_data_integrity=False)
    events_array = np.zeros((data['x'].shape[0], 4), np.float32)
    events_array[:, 0] = data['x'][:, 0]
    events_array[:, 1] = data['y'][:, 0]
    events_array[:, 2] = data['ts'][:, 0]
    events_array[:, 3] = data['p'][:, 0]
    return events_array


def translate_pointcloud(pc):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=0.2, high=0.2, size=[3])
    translated_pc = np.add(np.multiply(pc, xyz1), xyz2)  # .astype('float32')
    return translated_pc


def random_select_events(events, range=[0.95, 1]):
    t_max = events[-1, 2]
    time = random.randint(int(range[0] * t_max), int(range[1] * t_max))
    t_start = random.randint(0, int(t_max - time))
    t_end = t_start + time
    beg = binary_search_numpy_array(events[:, 2], 0, len(events[:, 2]) - 1, t_start)
    end = binary_search_numpy_array(events[:, 2], 0, len(events[:, 2]) - 1, t_end)
    events_selected = events[beg:end]
    if events_selected.shape[0] <= 1:
        return events
    else:
        events_selected[:, 2] = events_selected[:, 2] - events_selected[0, 2]
        return events_selected


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    return events


def jitter_pointcloud(events, sigma=0.01, clip=0.02):
    events_aug = events.clone()
    N, C = events_aug.shape
    rdn_value = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    events_aug[:, :3] += rdn_value
    return events_aug


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
    return events


def random_time_flip(event_tensor, p=0.5):
    if np.random.random() < p:
        event_tensor = np.flip(event_tensor, axis=0)
        event_tensor[:, 2] = event_tensor[0, 2] - event_tensor[:, 2]
        event_tensor[:, 3] = - event_tensor[:, 3]
    return event_tensor


def get_ev_voxel_grid(evArray, channelNum, resolution, voxel_size):
    """
    :param evArray: a [4 x N] Numpy array containing one event per row in the form:[x, y, timestamp, pol]
    :param channelNum: blocks' number of voxel grid form of event array.
    :param resolution: default(180, 240)
    """
    assert (evArray.shape[0] == 4)
    evArray = evArray.copy()
    if len(evArray[0]) != 0 and len(evArray[1]) != 0:
        if resolution:
            width = resolution[1]
            height = resolution[0]
        else:
            width = int(np.max(evArray[0, :]))
            width = width + voxel_size[1] - width % voxel_size[1]
            height = int(np.max(evArray[1, :]))
            height = height + voxel_size[0] - height % voxel_size[0]

        evFrame = np.zeros((channelNum, height, width), np.float32).ravel()
        # evCount = np.zeros((channelNum, height, width), np.float32).ravel()
    else:
        return torch.zeros((channelNum, 224, 224)).float(), torch.zeros((channelNum, 224, 224)).float()

    last_stamp = evArray[2, -1]
    first_stamp = evArray[2, 0]
    delta_t = last_stamp - first_stamp

    if delta_t == 0:
        delta_t = 1.0

    evArray[2, :] = (channelNum) * (evArray[2, :] - first_stamp) / delta_t
    x = evArray[0].astype(np.int)
    y = evArray[1].astype(np.int)
    # x = evArray[0]
    # y = evArray[1]
    ts = evArray[2]
    p = evArray[3]
    np.seterr(invalid='ignore')
    p[p == 2] = -1
    p[p == 0] = -1
    tis_c = ts.astype(np.int)
    # tis_c = ts
    tis_c[tis_c < 0] = 0
    dts = ts - tis_c
    """integrate events using single-side interpolation"""
    vals_left = p * (1.0 - dts)
    vals_right = p * dts
    valid_indices = tis_c < channelNum
    np.add.at(evFrame, x[valid_indices] + y[valid_indices] * width + tis_c[valid_indices] * width * height,
              vals_right[valid_indices])
    # np.add.at(evCount, x[valid_indices] + y[valid_indices] * width
    #           + tis_c[valid_indices] * width * height, 1)
    valid_indices = tis_c > 0
    np.add.at(evFrame, x[valid_indices] + y[valid_indices] * width
              + (tis_c[valid_indices] - 1) * width * height, vals_left[valid_indices])

    """"""
    evFrame = np.reshape(evFrame, (channelNum, height, width))
    # evCount = np.reshape(evCount, (channelNum, height, width))

    evFrame = torch.from_numpy(evFrame)
    # evCount = torch.from_numpy(evCount)

    # return evCount, evFrame
    return evFrame


def get_event_patch(x, ev_feats, patch_size, n_sample, output_size):
    b, c, h, w = x.shape
    x = (x.unfold(1, 1, 1).unfold(2, patch_size, patch_size).unfold(3, patch_size,
                                                                    patch_size)).contiguous()
    x = x.view(b, -1, patch_size, patch_size)
    """density selection"""
    x_sum = torch.flatten(x, start_dim=2).sum(dim=-1, )
    if x_sum.shape[1] > n_sample:
        values, idx = torch.topk(x_sum, k=n_sample, dim=-1, largest=True, sorted=False)
    elif x_sum.shape[1] <= n_sample:
        idx = torch.linspace(start=0, end=x_sum.shape[1] - 1, steps=x_sum.shape[1]).long().unsqueeze(0).repeat(b, 1)
        n_sample = x_sum.shape[1]
    """map idx to position"""
    total_patch_per_frame = torch.div(h, patch_size, rounding_mode='trunc') * torch.div(w, patch_size,
                                                                                        rounding_mode='trunc')
    idx_position_t = torch.div(idx, total_patch_per_frame, rounding_mode='trunc')
    idx_position_h = (idx - torch.div(idx, total_patch_per_frame, rounding_mode='trunc') * total_patch_per_frame) % (
        torch.div(w, patch_size, rounding_mode='trunc'))
    idx_position_w = torch.div(
        (idx - torch.div(idx, total_patch_per_frame, rounding_mode='trunc') * total_patch_per_frame),
        (torch.div(w, patch_size, rounding_mode='trunc')), rounding_mode='trunc')
    patch_position = torch.stack([idx_position_h, idx_position_w, idx_position_t], dim=1).float()
    """map idx to 2D features"""
    ev_feats_flat = (ev_feats.unfold(1, 1, 1).unfold(2, patch_size, patch_size)
                     .unfold(3, patch_size, patch_size)
                     .contiguous())
    ev_feats_flat = ev_feats_flat.view(x.shape[0], -1, patch_size, patch_size)
    patch_feats = index_patch(ev_feats_flat, idx)  # b, n_sample, patch_size, patch_size
    """pooling feature of patches to output size"""
    AvgPooling = torch.nn.AdaptiveAvgPool2d(output_size)
    patch_feats = AvgPooling(patch_feats)
    """"""
    patch_feats = patch_feats.view(b, n_sample, output_size ** 2)
    return patch_position, patch_feats, idx


def get_event_patch_sorted(x, ev_feats, patch_size, n_sample, output_size):
    b, c, h, w = x.shape
    x = (x.unfold(1, 1, 1).unfold(2, patch_size, patch_size).unfold(3, patch_size,
                                                                    patch_size)).contiguous()
    x = x.view(b, -1, patch_size, patch_size)
    """density selection"""
    x_sum = torch.flatten(x, start_dim=2).sum(dim=-1, )
    if x_sum.shape[1] > n_sample:
        values, idx = torch.topk(x_sum, k=n_sample, dim=-1, largest=True, sorted=True)
    elif x_sum.shape[1] <= n_sample:
        idx = torch.linspace(start=0, end=x_sum.shape[1] - 1, steps=x_sum.shape[1]).long().unsqueeze(0).repeat(b, 1)
        n_sample = x_sum.shape[1]
    """map idx to position"""
    total_patch_per_frame = torch.div(h, patch_size, rounding_mode='trunc') * torch.div(w, patch_size,
                                                                                        rounding_mode='trunc')
    idx_position_t = torch.div(idx, total_patch_per_frame, rounding_mode='trunc')
    idx_position_h = (idx - torch.div(idx, total_patch_per_frame, rounding_mode='trunc') * total_patch_per_frame) % (
        torch.div(w, patch_size, rounding_mode='trunc'))
    idx_position_w = torch.div(
        (idx - torch.div(idx, total_patch_per_frame, rounding_mode='trunc') * total_patch_per_frame),
        (torch.div(w, patch_size, rounding_mode='trunc')), rounding_mode='trunc')
    patch_position = torch.stack([idx_position_h, idx_position_w, idx_position_t], dim=1).float()
    """map idx to 2D features"""
    ev_feats_flat = (ev_feats.unfold(1, 1, 1).unfold(2, patch_size, patch_size)
                     .unfold(3, patch_size, patch_size)
                     .contiguous())
    ev_feats_flat = ev_feats_flat.view(x.shape[0], -1, patch_size, patch_size)
    patch_feats = index_patch(ev_feats_flat, idx)  # b, n_sample, patch_size, patch_size
    """pooling feature of patches to output size"""
    AvgPooling = torch.nn.AdaptiveAvgPool2d(output_size)
    patch_feats = AvgPooling(patch_feats)
    """"""
    patch_feats = patch_feats.view(b, n_sample, output_size ** 2)
    return patch_position, patch_feats, idx


def get_ev_voxel(event_ary, channel_num, patch_size, n_sample, output_size, resolution=None):
    event_ary = event_ary.T
    evCount, evFrame = get_ev_voxel_grid(event_ary, channelNum=channel_num,
                                         voxel_size=[patch_size, patch_size],
                                         resolution=resolution)
    evCount, evFrame = evCount.unsqueeze(0), evFrame.unsqueeze(0)
    # patch_position, patch_feats, idx = get_event_patch(evCount, ev_feats=evFrame,
    #                                                    patch_size=patch_size, n_sample=n_sample,
    #                                                    output_size=output_size)

    patch_position, patch_feats, idx = get_event_patch_sorted(evCount, ev_feats=evFrame,
                                                              patch_size=patch_size, n_sample=n_sample,
                                                              output_size=output_size)

    """Refined version"""
    filter_idx = (patch_feats.sum(2) != 0).nonzero(as_tuple=True)[1]
    if len(filter_idx) <= 1:
        return patch_position.squeeze().transpose(0, 1), patch_feats.squeeze(), idx, evFrame
    patch_position_filter = patch_position[:, :, filter_idx].squeeze().transpose(0, 1)
    patch_feats_filter = patch_feats[:, filter_idx].squeeze()
    return patch_position_filter, patch_feats_filter, idx, evFrame


# -------------------------------------------------------------------------------------------------------------------- #
"""IO stream tools"""


# -------------------------------------------------------------------------------------------------------------------- #
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def events_to_image(xs, ys, ps, sensor_size):
    """
    Accumulate events into an image.
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def events_to_channels(event: EventSequence):
    """
    Generate a two-channel event image containing event counters.
    """
    xs = event._features_torch[:, 0]
    ys = event._features_torch[:, 1]
    ps = event._features_torch[:, 3]
    sensor_size = [event._image_height, event._image_width]
    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])


def custom_erode(mask, kernel_size=3):
    kernel = torch.ones((1, 1, kernel_size, kernel_size))
    kernel[0, 0, kernel_size // 2, kernel_size // 2] = 0

    neighbors_diff = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)

    change_to_1 = (mask == 0) & (neighbors_diff >= 5)
    change_to_0 = (mask == 1) & (neighbors_diff <= 1)

    mask[change_to_1] = 1
    mask[change_to_0] = 0

    return mask


def dilate(mask, kernel_size=3):
    kernel = torch.ones((1, 1, kernel_size, kernel_size))
    dilated = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    dilated = (dilated > 0).float()

    return dilated


def process_mask(mask, erode_kernel_size=3, dilate_kernel_size=7):
    mask = torch.sum(mask, dim=0, keepdim=True)
    mask[mask > 0] = 1
    eroded_mask = custom_erode(mask.clone(), erode_kernel_size)
    dilated_mask = dilate(eroded_mask, dilate_kernel_size)
    return dilated_mask


def make_different_map(img1, img2):
    difference_map = img1 - img2  # [B, 3, H, W], difference map on IB
    difference_map = torch.sum(torch.abs(difference_map), dim=1, keepdim=True)  # [B, 1, H, W]
    return difference_map


# Function to divide the difference map into patches
def divide_into_patches(difference_map, patch_size=32):
    B, C, H, W = difference_map.shape
    patches = difference_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)  # [B, C, num_patches, patch_size, patch_size]
    return patches


def calculate_patch_sums(patches):
    patch_sums = patches.sum(dim=[3, 4])  # [B, C, num_patches]
    return patch_sums


def modify_patches_based_on_sums(patches, patch_sums, beta):
    B, C, num_patches, patch_size, _ = patches.shape
    top_patches_mask = patch_sums >= beta
    patches = patches.view(B, C, num_patches, patch_size, patch_size)
    patches[~top_patches_mask] = 0
    patches[top_patches_mask] = 1

    return patches.view(B, C, num_patches, patch_size, patch_size)


def modify_patches_based_on_one_or_zero(patches, patch_sums):
    B, C, num_patches, patch_size, _ = patches.shape
    patches_mask = torch.zeros_like(patch_sums.view(B, -1), dtype=torch.bool)
    patches_mask[patch_sums.squeeze(1) > 0] = True

    patches_mask = patches_mask.view(B, C, num_patches)
    patches = patches.view(B, C, num_patches, patch_size, patch_size)
    patches[~patches_mask] = 0
    patches[patches_mask] = 1

    return patches.view(B, C, num_patches, patch_size, patch_size)


# Reconstruct the modified difference map
def reconstruct_from_patches(patches, image_size):
    B, C, num_patches, patch_size, _ = patches.shape
    num_patches_per_row = image_size[1] // patch_size
    num_patches_per_col = image_size[0] // patch_size

    patches = patches.view(B, C, num_patches_per_col, num_patches_per_row, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    reconstructed = patches.view(B, C, image_size[0], image_size[1])
    return reconstructed


def pyramid_patch_mask(img0, img1, event_mask, patch_size=32, beta=0.1):
    difference_map = make_different_map(img0, img1)
    patches = divide_into_patches(difference_map, patch_size)
    patch_sums = calculate_patch_sums(patches)
    modified_patches = modify_patches_based_on_sums(patches, patch_sums, beta)
    modified_difference_map = reconstruct_from_patches(modified_patches, difference_map.shape[2:])

    event_mask_patched = generate_event_mask_patch_mask(event_mask, patch_size)
    modified_difference_map = modified_difference_map * event_mask_patched

    dm_0 = modified_difference_map
    dm_1 = down_sample_mask(modified_difference_map)
    dm_2 = down_sample_mask(dm_1)
    return [dm_0, dm_1, dm_2]


def mask_reverse(mask1, mask2):
    new_mask1 = mask1.clone()
    new_mask2 = mask2.clone()

    swap_condition = new_mask1 != new_mask2
    temp = new_mask1[swap_condition].clone()
    new_mask1[swap_condition] = new_mask2[swap_condition]
    new_mask2[swap_condition] = temp

    return new_mask1, new_mask2


def generate_event_mask_patch_mask(event_mask, patch_size=32):
    patches = divide_into_patches(event_mask, patch_size)
    patch_sums = calculate_patch_sums(patches)
    modified_patches = modify_patches_based_on_one_or_zero(patches, patch_sums)
    modified_event_patches_map = reconstruct_from_patches(modified_patches, event_mask.shape[2:])
    return modified_event_patches_map


def shrink_mask(event_mask, patch_size):
    mask = generate_event_mask_patch_mask(event_mask, patch_size)
    b, _, height, width = mask.size()
    new_patch_size = int(patch_size / 2)

    patches = mask.view(b, 1, height // patch_size, patch_size, width // patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(b, 1, -1, patch_size, patch_size)

    pooled_patches, _ = torch.max(patches.view(b, 1, -1, 2, new_patch_size, 2, new_patch_size), dim=3, keepdim=True)
    pooled_patches, _ = torch.max(pooled_patches, dim=5, keepdim=True)

    new_height = height // 2
    new_width = width // 2
    new_mask = pooled_patches.view(b, 1, new_height, new_width)

    return new_mask


def down_sample_mask(mask):
    downsampled_tensor = F.avg_pool2d(mask, kernel_size=2)
    downsampled_tensor = torch.round(downsampled_tensor)
    return downsampled_tensor


def pyramid_Img(Img):
    img_pyr = [Img]
    for i in range(1, 3):
        img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
    return img_pyr
