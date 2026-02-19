# -*- coding: utf-8 -*-
import argparse
from typing import List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _tensor_to_int_list(tensor_proto):
    arr = numpy_helper.to_array(tensor_proto)
    return [int(v) for v in np.array(arr).reshape(-1).tolist()]


def _build_const_value_map(graph):
    const_map = {}
    for init in graph.initializer:
        const_map[init.name] = _tensor_to_int_list(init)

    for node in graph.node:
        if node.op_type != 'Constant' or len(node.output) != 1:
            continue
        value_attr = None
        for attr in node.attribute:
            if attr.name == 'value':
                value_attr = attr
                break
        if value_attr is None:
            continue
        const_map[node.output[0]] = _tensor_to_int_list(value_attr.t)
    return const_map


def _parse_slice_axis1(node, const_map):
    if node.op_type != 'Slice' or len(node.input) < 5 or len(node.output) != 1:
        return None
    starts_name, ends_name, axes_name, steps_name = node.input[1:5]
    if starts_name not in const_map or ends_name not in const_map:
        return None
    if axes_name not in const_map or steps_name not in const_map:
        return None

    starts = const_map[starts_name]
    ends = const_map[ends_name]
    axes = const_map[axes_name]
    steps = const_map[steps_name]
    if len(starts) != 1 or len(ends) != 1 or len(axes) != 1 or len(steps) != 1:
        return None
    if axes[0] != 1 or steps[0] != 1:
        return None

    start, end = int(starts[0]), int(ends[0])
    if end <= start:
        return None
    return {'start': start, 'end': end, 'output': node.output[0]}


def _parse_ranges(ranges_text: str) -> List[Tuple[int, int]]:
    if not ranges_text:
        return []
    ranges = []
    for token in ranges_text.split(','):
        token = token.strip()
        if not token:
            continue
        s, e = token.split(':')
        ranges.append((int(s), int(e)))
    return ranges


def convert_onnx_to_emblayer_joint(
    model,
    vec_ranges: List[Tuple[int, int]],
    seq_ranges: List[Tuple[int, int]],
    domain='com.deepctr',
    version=1,
    vec_pad_value=0.0,
    seq_pad_value=0,
):
    graph = model.graph
    const_map = _build_const_value_map(graph)

    vec_range_set = set(vec_ranges)
    seq_range_set = set(seq_ranges)
    if vec_range_set & seq_range_set:
        raise ValueError('vec_ranges and seq_ranges must not overlap')

    need_vec_inputs = bool(vec_ranges)
    need_seq_inputs = bool(seq_ranges)

    vec_values_name = 'emblayer_vec_values'
    vec_prefix_name = 'emblayer_vec_prefix'
    vec_indices_name = 'emblayer_vec_feat_indices'

    seq_values_name = 'emblayer_seq_values'
    seq_prefix_name = 'emblayer_seq_prefix'
    seq_lengths_name = 'emblayer_seq_lengths'

    existed_inputs = {x.name for x in graph.input}
    if need_vec_inputs:
        if vec_values_name not in existed_inputs:
            graph.input.extend([
                helper.make_tensor_value_info(vec_values_name, TensorProto.FLOAT, [None]),
                helper.make_tensor_value_info(vec_prefix_name, TensorProto.INT64, [None]),
                helper.make_tensor_value_info(vec_indices_name, TensorProto.INT64, [None]),
            ])
            existed_inputs.update({vec_values_name, vec_prefix_name, vec_indices_name})

    if need_seq_inputs:
        if seq_values_name not in existed_inputs:
            graph.input.extend([
                helper.make_tensor_value_info(seq_values_name, TensorProto.INT64, [None]),
                helper.make_tensor_value_info(seq_prefix_name, TensorProto.INT64, [None]),
                helper.make_tensor_value_info(seq_lengths_name, TensorProto.INT64, [None]),
            ])

    matched_vec = {}
    matched_seq = {}
    nodes_to_remove = set()
    for idx, node in enumerate(graph.node):
        parsed = _parse_slice_axis1(node, const_map)
        if parsed is None:
            continue
        cur_range = (parsed['start'], parsed['end'])
        if cur_range in vec_range_set and cur_range not in matched_vec:
            matched_vec[cur_range] = (idx, parsed['output'])
            nodes_to_remove.add(idx)
        elif cur_range in seq_range_set and cur_range not in matched_seq:
            matched_seq[cur_range] = (idx, parsed['output'])
            nodes_to_remove.add(idx)

    new_nodes = []
    removed_slice_nodes = len(nodes_to_remove)
    inserted_vec_nodes = 0
    inserted_seq_nodes = 0

    if matched_vec:
        ordered_vec_ranges = [r for r in vec_ranges if r in matched_vec]
        vec_outputs = [matched_vec[r][1] for r in ordered_vec_ranges]
        vec_starts = [int(r[0]) for r in ordered_vec_ranges]
        vec_ends = [int(r[1]) for r in ordered_vec_ranges]
        vec_widths = [int(r[1] - r[0]) for r in ordered_vec_ranges]
        vec_node = helper.make_node(
            'EmblayerVec',
            inputs=[vec_values_name, vec_prefix_name, vec_indices_name],
            outputs=vec_outputs,
            domain=domain,
            name='EmblayerVec_aggregated',
            output_dim=int(sum(vec_widths)),
            col_start=int(min(vec_starts) if vec_starts else 0),
            col_end=int(max(vec_ends) if vec_ends else 0),
            pad_value=float(vec_pad_value),
            vec_starts=vec_starts,
            vec_ends=vec_ends,
        )
        new_nodes.append(vec_node)
        inserted_vec_nodes = 1

    if matched_seq:
        ordered_seq_ranges = [r for r in seq_ranges if r in matched_seq]
        seq_outputs = [matched_seq[r][1] for r in ordered_seq_ranges]
        seq_starts = [int(r[0]) for r in ordered_seq_ranges]
        seq_ends = [int(r[1]) for r in ordered_seq_ranges]
        seq_widths = [int(r[1] - r[0]) for r in ordered_seq_ranges]
        seq_node = helper.make_node(
            'EmblayerSeq',
            inputs=[seq_values_name, seq_prefix_name, seq_lengths_name],
            outputs=seq_outputs,
            domain=domain,
            name='EmblayerSeq_aggregated',
            pad_value=int(seq_pad_value),
            max_seq_num=int(len(ordered_seq_ranges)),
            max_seq_len=int(max(seq_widths) if seq_widths else 0),
            output_2d=1,
            seq_starts=seq_starts,
            seq_ends=seq_ends,
        )
        new_nodes.append(seq_node)
        inserted_seq_nodes = 1

    for idx, node in enumerate(graph.node):
        if idx in nodes_to_remove:
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    has_custom_domain = any(op.domain == domain for op in model.opset_import)
    if not has_custom_domain:
        model.opset_import.append(helper.make_opsetid(domain, version))

    return model, {
        'removed_slice_nodes': removed_slice_nodes,
        'inserted_vec_nodes': inserted_vec_nodes,
        'inserted_seq_nodes': inserted_seq_nodes,
    }


def main():
    parser = argparse.ArgumentParser(description='Joint converter: Slice -> EmblayerVec/EmblayerSeq')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vec_ranges', type=str, default='')
    parser.add_argument('--seq_ranges', type=str, default='')
    parser.add_argument('--domain', type=str, default='com.deepctr')
    parser.add_argument('--domain_version', type=int, default=1)
    parser.add_argument('--vec_pad_value', type=float, default=0.0)
    parser.add_argument('--seq_pad_value', type=int, default=0)
    args = parser.parse_args()

    vec_ranges = _parse_ranges(args.vec_ranges)
    seq_ranges = _parse_ranges(args.seq_ranges)
    if not vec_ranges and not seq_ranges:
        raise ValueError('at least one of --vec_ranges/--seq_ranges must be provided')

    model = onnx.load(args.input)
    model, stats = convert_onnx_to_emblayer_joint(
        model,
        vec_ranges=vec_ranges,
        seq_ranges=seq_ranges,
        domain=args.domain,
        version=args.domain_version,
        vec_pad_value=args.vec_pad_value,
        seq_pad_value=args.seq_pad_value,
    )
    onnx.save(model, args.output)

    print('Convert done!')
    print('input:', args.input)
    print('output:', args.output)
    print('vec_ranges:', vec_ranges)
    print('seq_ranges:', seq_ranges)
    print('removed Slice nodes:', stats['removed_slice_nodes'])
    print('inserted EmblayerVec nodes:', stats['inserted_vec_nodes'])
    print('inserted EmblayerSeq nodes:', stats['inserted_seq_nodes'])


if __name__ == '__main__':
    main()
