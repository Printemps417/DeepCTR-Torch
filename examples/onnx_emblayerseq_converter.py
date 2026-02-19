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


def _parse_seq_ranges(seq_ranges: str) -> List[Tuple[int, int]]:
    items = []
    for token in seq_ranges.split(','):
        token = token.strip()
        if not token:
            continue
        s, e = token.split(':')
        items.append((int(s), int(e)))
    return items


def convert_onnx_to_emblayerseq(
    model,
    seq_ranges: List[Tuple[int, int]],
    domain='com.deepctr',
    version=1,
    pad_value=0,
):
    graph = model.graph
    const_map = _build_const_value_map(graph)

    matched = {}
    nodes_to_remove = set()

    for idx, node in enumerate(graph.node):
        parsed = _parse_slice_axis1(node, const_map)
        if parsed is None:
            continue
        cur_range = (parsed['start'], parsed['end'])
        if cur_range not in seq_ranges:
            continue
        if cur_range in matched:
            continue
        matched[cur_range] = (idx, parsed['output'])
        nodes_to_remove.add(idx)

    existed_inputs = {x.name for x in graph.input}
    values_name = 'emblayer_seq_values'
    prefix_name = 'emblayer_seq_prefix'
    lengths_name = 'emblayer_seq_lengths'
    if values_name not in existed_inputs:
        graph.input.extend([
            helper.make_tensor_value_info(values_name, TensorProto.INT64, [None]),
            helper.make_tensor_value_info(prefix_name, TensorProto.INT64, [None]),
            helper.make_tensor_value_info(lengths_name, TensorProto.INT64, [None]),
        ])

    new_nodes = []
    inserted = 0
    if matched:
        ordered_ranges = [r for r in seq_ranges if r in matched]
        ordered_outputs = [matched[r][1] for r in ordered_ranges]
        seq_starts = [int(r[0]) for r in ordered_ranges]
        seq_ends = [int(r[1]) for r in ordered_ranges]
        seq_widths = [int(r[1] - r[0]) for r in ordered_ranges]

        seq_node = helper.make_node(
            'EmblayerSeq',
            inputs=[values_name, prefix_name, lengths_name],
            outputs=ordered_outputs,
            domain=domain,
            name='EmblayerSeq_aggregated',
            pad_value=int(pad_value),
            max_seq_num=int(len(ordered_ranges)),
            max_seq_len=int(max(seq_widths) if seq_widths else 0),
            output_2d=1,
            seq_starts=seq_starts,
            seq_ends=seq_ends,
        )
        new_nodes.append(seq_node)
        inserted = 1

    for idx, node in enumerate(graph.node):
        if idx not in nodes_to_remove:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    has_custom_domain = any(op.domain == domain for op in model.opset_import)
    if not has_custom_domain:
        model.opset_import.append(helper.make_opsetid(domain, version))

    return model, len(nodes_to_remove), inserted


def main():
    parser = argparse.ArgumentParser(
        description='Convert sequence Slice nodes into custom EmblayerSeq nodes (compressed seq format)'
    )
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seq_ranges', required=True, help='comma ranges, e.g. 5:9,10:14')
    parser.add_argument('--domain', type=str, default='com.deepctr')
    parser.add_argument('--domain_version', type=int, default=1)
    parser.add_argument('--pad_value', type=int, default=0)
    args = parser.parse_args()

    ranges = _parse_seq_ranges(args.seq_ranges)
    model = onnx.load(args.input)
    model, removed, inserted = convert_onnx_to_emblayerseq(
        model,
        seq_ranges=ranges,
        domain=args.domain,
        version=args.domain_version,
        pad_value=args.pad_value,
    )
    onnx.save(model, args.output)

    print('Convert done!')
    print('input:', args.input)
    print('output:', args.output)
    print('seq_ranges:', ranges)
    print('removed Slice nodes:', removed)
    print('inserted EmblayerSeq nodes:', inserted)


if __name__ == '__main__':
    main()
