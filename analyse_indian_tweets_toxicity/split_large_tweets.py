import argparse
import sys

def split_file(source, dest_folder, num_files):
    smallfile = None
    num_lines = sum(1 for line in open(source))
    lines_per_file = num_lines / num_files
    print(num_lines)
    with open(source) as bigfile:
        file_num = 0
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    print(f"Closing {small_filename}")
                    smallfile.close()
                small_filename = f'{dest_folder}/small_file_{file_num}.txt'
                smallfile = open(small_filename, "w")
                file_num += 1
            smallfile.write(line)
        if smallfile:
            smallfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        type=str,
        help="File containing lots of test data",
    )
    parser.add_argument(
        "-d",
        "--dest",
        default=None,
        type=str,
        help="Where to save split up file",
    )
    parser.add_argument(
        "-f",
        "--files",
        default=None,
        type=int,
        help="Number of files to split into",
    )
    args = parser.parse_args()

    if args.source is None or args.dest is None or args.files is None:
        print("No arguments given")
        sys.exit()

    print(f"Splitting file into {args.files} files...")

    split_file(args.source, args.dest, args.files)

    print("Finished.")
    print("Starting toxicity analysis...")