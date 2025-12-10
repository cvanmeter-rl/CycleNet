from cleanfid import fid

REAL_DIR = "/mnt/project/data/real/all"
SIM_DIR = "/mnt/project/data/sim/all"


def main():
    score = fid.compute_fid(REAL_DIR, SIM_DIR)
    print(f"FID Score: {score}")

if __name__ == "__main__":
    main()