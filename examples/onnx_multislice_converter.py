# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper


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


def _parse_slice_node(node, const_map):
    if node.op_type != 'Slice' or len(node.input) < 5 or len(node.output) != 1:
        return None

    data_name = node.input[0]
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

    start = int(starts[0])
    end = int(ends[0])
    if end <= start:
        return None

    return {
        'data_input': data_name,
        'start': start,
        'length': end - start,
        'output': node.output[0],
    }


def convert_onnx_to_multislice(
    model,
    min_group_size=2,
    domain='com.deepctr',
    version=1,
    include_col_start=None,
    include_col_end=None,
):
    graph = model.graph
    const_map = _build_const_value_map(graph)

    parsed_by_index = {}
    groups = defaultdict(list)

    for idx, node in enumerate(graph.node):
        parsed = _parse_slice_node(node, const_map)
        if parsed is None:
            continue
        slice_end = parsed['start'] + parsed['length']
        if include_col_start is not None and parsed['start'] < include_col_start:
            continue
        if include_col_end is not None and slice_end > include_col_end:
            continue
        parsed_by_index[idx] = parsed
        groups[parsed['data_input']].append((idx, parsed))

    valid_groups = []
    for data_input, items in groups.items():
        if len(items) < min_group_size:
            continue
        items = sorted(items, key=lambda x: x[0])
        valid_groups.append((data_input, items))

    if not valid_groups:
        return model, 0, 0

    slice_indices_to_remove = set()
    group_first_index = {}
    group_meta = {}

    for data_input, items in valid_groups:
        first_idx = items[0][0]
        group_first_index[first_idx] = data_input
        starts = [item[1]['start'] for item in items]
        lengths = [item[1]['length'] for item in items]
        outputs = [item[1]['output'] for item in items]
        for idx, _ in items:
            slice_indices_to_remove.add(idx)
        group_meta[data_input] = {
            'starts': starts,
            'lengths': lengths,
            'outputs': outputs,
        }

    new_nodes = []
    inserted_multislice_nodes = 0

    for idx, node in enumerate(graph.node):
        if idx in group_first_index:
            data_input = group_first_index[idx]
            meta = group_meta[data_input]
            multislice_node = helper.make_node(
                'MultiSlice',
                inputs=[data_input],
                outputs=meta['outputs'],
                domain=domain,
                starts=meta['starts'],
                lengths=meta['lengths'],
                name=f'MultiSlice_{inserted_multislice_nodes}',
            )
            new_nodes.append(multislice_node)
            inserted_multislice_nodes += 1

        if idx in slice_indices_to_remove:
            continue

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    has_custom_domain = any(op.domain == domain for op in model.opset_import)
    if not has_custom_domain:
        model.opset_import.append(helper.make_opsetid(domain, version))

    removed_slice_nodes = len(slice_indices_to_remove)
    return model, removed_slice_nodes, inserted_multislice_nodes


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX Slice groups to custom MultiSlice nodes')
    parser.add_argument('--input', required=True, help='Input ONNX path')
    parser.add_argument('--output', required=True, help='Output ONNX path')
    parser.add_argument('--min_group_size', type=int, default=2)
    parser.add_argument('--domain', type=str, default='com.deepctr')
    parser.add_argument('--domain_version', type=int, default=1)
    parser.add_argument('--include_col_start', type=int, default=None)
    parser.add_argument('--include_col_end', type=int, default=None)
    args = parser.parse_args()

    model = onnx.load(args.input)
    model, removed, inserted = convert_onnx_to_multislice(
        model,
        min_group_size=args.min_group_size,
        domain=args.domain,
        version=args.domain_version,
        include_col_start=args.include_col_start,
        include_col_end=args.include_col_end,
    )
    onnx.save(model, args.output)

    print('Convert done!')
    print('input:', args.input)
    print('output:', args.output)
    print('removed Slice nodes:', removed)
    print('inserted MultiSlice nodes:', inserted)


if __name__ == '__main__':
    main()
