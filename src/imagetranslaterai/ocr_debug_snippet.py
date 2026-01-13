
            # Debugging result structure to file
            with open("debug_ocr.txt", "w", encoding="utf-8") as f:
                f.write(f"Type: {type(result)}\n")
                if result:
                    f.write(f"Result[0] type: {type(result[0])}\n")
                    f.write(f"Result content: {result}\n")
