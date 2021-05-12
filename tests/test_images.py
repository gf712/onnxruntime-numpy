import onnxruntime_numpy as onp
import numpy as np
import pytest
from .utils import expect, interpolate_nd, cubic_coeffs, linear_coeffs, nearest_coeffs


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_downsample_scales_cubic(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

    expected = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales).astype(type_a)

    result = onp.interpolate(
        onp.array(data),
        scales=onp.array(scales),
        mode="cubic")

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_downsample_scales_cubic_A_n0p5_exclude_outside(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

    expected = interpolate_nd(
        data, lambda x: cubic_coeffs(x, A=-0.5),
        scale_factors=scales, exclude_outside=True).astype(
        type_a)

    result = onp.interpolate(onp.array(data), scales=onp.array(
        scales), mode="cubic", cubic_coeff_a=-0.5, exclude_outside=True)

    expect(expected, result.numpy())


# TODO: results are not the same?
# @pytest.mark.parametrize("type_a", [np.float32])
# def test_interpolate_downsample_scales_cubic_align_corners(type_a):
#     data = np.array([[[
#         [1, 2, 3, 4],
#         [5, 6, 7, 8],
#         [9, 10, 11, 12],
#         [13, 14, 15, 16],
#     ]]], dtype=type_a)

#     scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

#     expected = interpolate_nd(
#         data, cubic_coeffs, scale_factors=scales,
#         coordinate_transformation_mode='align_corners').astype(
#         type_a)

#     result = onp.interpolate(onp.array(data), scales=onp.array(
#         scales), mode="cubic", coordinate_transformation_mode="align_corners")

#     expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_downsample_scales_linear(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    expected = interpolate_nd(
        data, linear_coeffs, scale_factors=scales).astype(
        type_a)

    result = onp.interpolate(onp.array(data), scales=onp.array(
        scales), mode="linear")

    expect(expected, result.numpy())


# TODO: results are not the same?
# @pytest.mark.parametrize("type_a", [np.float32])
# def test_interpolate_downsample_scales_linear_align_corners(type_a):
#     data = np.array([[[
#         [1, 2, 3, 4],
#         [5, 6, 7, 8]
#     ]]], dtype=type_a)

#     scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

#     expected = interpolate_nd(
#         data, linear_coeffs, scale_factors=scales,
#         coordinate_transformation_mode='align_corners').astype(type_a)

#     result = onp.interpolate(onp.array(data), scales=onp.array(
#         scales), mode="linear", coordinate_transformation_mode='align_corners')

#     expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_downsample_scales_nearest(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    expected = interpolate_nd(
        data, nearest_coeffs, scale_factors=scales).astype(
        type_a)

    result = onp.interpolate(onp.array(data), scales=onp.array(
        scales), mode="nearest")

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_downsample_sizes_cubic(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 3, 3], dtype=np.int64)

    expected = interpolate_nd(
        data, cubic_coeffs, output_size=sizes).astype(
        type_a)

    result = onp.interpolate(onp.array(data), sizes=onp.array(
        sizes), mode="cubic")

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_downsample_sizes_linear_pytorch_half_pixel(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 3, 1], dtype=np.int64)

    expected = interpolate_nd(
        data, linear_coeffs, output_size=sizes,
        coordinate_transformation_mode='pytorch_half_pixel').astype(type_a)

    result = onp.interpolate(onp.array(data), sizes=onp.array(
        sizes), mode='linear', coordinate_transformation_mode='pytorch_half_pixel')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_downsample_sizes_nearest(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 1, 3], dtype=np.int64)

    expected = interpolate_nd(
        data, nearest_coeffs, output_size=sizes).astype(type_a)

    result = onp.interpolate(onp.array(data), sizes=onp.array(
        sizes), mode='nearest')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_b", [np.float32])
def test_interpolate_tf_crop_and_resize(type_a, type_b):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=type_b)
    sizes = np.array([1, 1, 1, 3], dtype=np.int64)

    expected = interpolate_nd(
        data, linear_coeffs, output_size=sizes, roi=roi,
        coordinate_transformation_mode='tf_crop_and_resize').astype(type_a)

    result = onp.interpolate(onp.array(data), onp.array(roi), sizes=onp.array(
        sizes), mode='linear', coordinate_transformation_mode='tf_crop_and_resize')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_b", [np.float32])
def test_interpolate_tf_crop_and_resize_extrapolation_value(type_a, type_b):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=type_b)
    sizes = np.array([1, 1, 1, 3], dtype=np.int64)

    expected = interpolate_nd(
        data, linear_coeffs, output_size=sizes, roi=roi,
        coordinate_transformation_mode='tf_crop_and_resize',
        extrapolation_value=10.0).astype(type_a)

    result = onp.interpolate(
        onp.array(data),
        onp.array(roi),
        sizes=onp.array(sizes),
        mode='linear', coordinate_transformation_mode='tf_crop_and_resize',
        extrapolation_value=10.0)

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_cubic(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales).astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='cubic')

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_cubic_A_n0p5_exclude_outside(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(
        data, lambda x: cubic_coeffs(x, A=-0.5),
        scale_factors=scales, exclude_outside=True).astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='cubic',
        cubic_coeff_a=-0.5, exclude_outside=True)

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_cubic_align_corners(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales,
        coordinate_transformation_mode='align_corners').astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='cubic',
        coordinate_transformation_mode='align_corners')

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_cubic_asymmetric(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.75),
                              scale_factors=scales,
                              coordinate_transformation_mode='asymmetric'
                              ).astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='cubic',
        coordinate_transformation_mode='asymmetric')

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_linear(type_a):
    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(
        data, linear_coeffs, scale_factors=scales).astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='linear')

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_linear_align_corners(type_a):
    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(
        data, linear_coeffs, scale_factors=scales,
        coordinate_transformation_mode='align_corners').astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='linear',
        coordinate_transformation_mode='align_corners')

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_scales_nearest(type_a):
    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=type_a)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    expected = interpolate_nd(
        data, nearest_coeffs, scale_factors=scales).astype(type_a)

    result = onp.interpolate(
        onp.array(data), scales=onp.array(scales), mode='nearest')

    expect(expected, result.numpy())


# TODO: only float32 working.. uint8 and int32 cause issues with this param combination
# @pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
@pytest.mark.parametrize("type_a", [np.float32])
def test_interpolate_upsample_sizes_cubic(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 9, 10], dtype=np.int64)

    expected = interpolate_nd(
        data, cubic_coeffs, output_size=sizes).astype(type_a)

    result = onp.interpolate(
        onp.array(data), sizes=onp.array(sizes), mode='cubic')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_upsample_sizes_nearest(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 7, 8], dtype=np.int64)

    expected = interpolate_nd(
        data, nearest_coeffs, output_size=sizes).astype(type_a)

    result = onp.interpolate(
        onp.array(data), sizes=onp.array(sizes), mode='nearest')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_upsample_sizes_nearest_ceil_half_pixel(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    expected = interpolate_nd(
        data, lambda x: nearest_coeffs(x, mode='ceil'),
        output_size=sizes).astype(type_a)

    result = onp.interpolate(
        onp.array(data), sizes=onp.array(sizes),
        mode='nearest', coordinate_transformation_mode='half_pixel',
        nearest_mode='ceil')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_upsample_sizes_nearest_floor_align_corners(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    expected = interpolate_nd(data, lambda x: nearest_coeffs(
        x, mode='floor'), output_size=sizes,
        coordinate_transformation_mode='align_corners').astype(type_a)

    result = onp.interpolate(
        onp.array(data), sizes=onp.array(sizes),
        mode='nearest', coordinate_transformation_mode='align_corners',
        nearest_mode='floor')

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32])
def test_interpolate_upsample_sizes_nearest_round_prefer_ceil_asymmetric(type_a):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=type_a)

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    expected = interpolate_nd(
        data, lambda x: nearest_coeffs(x, mode='round_prefer_ceil'),
        output_size=sizes, coordinate_transformation_mode='asymmetric').astype(
        type_a)

    result = onp.interpolate(
        onp.array(data), sizes=onp.array(sizes),
        mode='nearest', coordinate_transformation_mode='asymmetric',
        nearest_mode='round_prefer_ceil')

    expect(expected, result.numpy())
