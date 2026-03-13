def hello_world(output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write("Hello, World!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello World")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    args = parser.parse_args()
    hello_world(args.output_path)