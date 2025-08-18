#!/usr/bin/env python3
import os, glob, csv, argparse, requests, mimetypes, sys

def find_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")
    paths = []
    for pat in exts:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(paths)

def open_batch(paths):
    """
    Build the 'files' list for requests.post(files=...) with proper per-file Content-Type.
    Returns (files_list, open_handles). Remember to close handles after the request.
    """
    handles = []
    try:
        files = []
        for p in paths:
            h = open(p, "rb")
            handles.append(h)
            ctype = mimetypes.guess_type(p)[0]
            if not ctype:
                ctype = "image/png" if p.lower().endswith(".png") else "image/jpeg"
            files.append(("files", (os.path.basename(p), h, ctype)))
        return files, handles
    except Exception:
        for h in handles:
            try: h.close()
            except: pass
        raise

def post_batch(url, files):
    r = requests.post(url, files=files, timeout=60)
    r.raise_for_status()  # will raise for 4xx/5xx
    return r.json()

def main():
    ap = argparse.ArgumentParser(description="Send a folder of images to /predict-batch and write CSV.")
    ap.add_argument("folder", nargs="?", default="samples", help="Folder with .jpg/.jpeg/.png (default: samples)")
    ap.add_argument("--api", default="http://127.0.0.1:8000/predict-batch", help="Batch endpoint URL")
    ap.add_argument("--out", default="folder_results.csv", help="Output CSV path")
    ap.add_argument("--batch", type=int, default=16, help="Batch size per POST (default: 16)")
    args = ap.parse_args()

    paths = find_images(args.folder)
    print(f"Found {len(paths)} images in {args.folder}")
    if not paths:
        return 1

    # Write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "score"])

        i = 0
        while i < len(paths):
            chunk = paths[i:i+args.batch]
            files, handles = open_batch(chunk)
            try:
                resp = post_batch(args.api, files)  # should be a list of {file,label,score,...}
            except requests.HTTPError as e:
                # show server response body to help debug
                body = getattr(e.response, "text", "")
                print(f"[HTTP {e.response.status_code}] {body}", file=sys.stderr)
                raise
            finally:
                for h in handles:
                    try: h.close()
                    except: pass

            # write rows
            for item in resp:
                w.writerow([item.get("file"), item.get("label"), item.get("score")])

            i += args.batch
            print(f"Processed {min(i, len(paths))}/{len(paths)}")

    print(f"Wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main() or 0)

