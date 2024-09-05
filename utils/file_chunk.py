import argparse
import os
import yaml

def split_file(file_path, chunk_size):
    part_num = 0
    parts = []
    base_dir = os.path.dirname(file_path)
    with open(file_path, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            part_num += 1
            part_name = f"{file_path}.part{part_num}"
            with open(part_name, 'wb') as part_file:
                part_file.write(chunk)
            parts.append(part_name)
            chunk = f.read(chunk_size)
    return parts

def combine_files(parts, output_filename):
    with open(output_filename, 'wb') as output_file:
        for part_name in parts:
            with open(part_name, 'rb') as part_file:
                output_file.write(part_file.read())


def split_mode(files, chunk_size):
    metadata = {'files': []}
    for file in files:
        parts = split_file(file, chunk_size)
        metadata['files'].append({
            'original': file,
            'parts': parts
        })

    # Write YAML file
    with open('metadata.yaml', 'w') as yaml_file:
        yaml.dump(metadata, yaml_file)

def combine_mode(yaml_file):
    with open(yaml_file, 'r') as file:
        metadata = yaml.safe_load(file)
        for file_info in metadata['files']:
            combine_files(file_info['parts'], file_info['original'])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Tool to split and combine files.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # add_subparsers 创建了一个子命令的容器。dest='command' 指定了一个属性名，这个名字会用来保存哪个子命令被触发了。required=True 表示命令行中必须有一个子命令。

    # Split command
    split_parser = subparsers.add_parser('split', help='Split files into smaller parts.')
    split_parser.add_argument('-s', '--size', type=int, default=26214400, help='Chunk size in bytes. Default is 26214400 (25 MB).')
    split_parser.add_argument('-fs', '--files', nargs='+', help='List of files to split.')

    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine files from a metadata file.')
    combine_parser.add_argument('yaml_file', help='YAML file containing metadata about the files to combine.')
    

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.command == 'split':
        split_mode(args.files, args.size)
    elif args.command == 'combine':
        combine_mode(args.yaml_file)

if __name__ == "__main__":
    main()
