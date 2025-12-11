from cleanfid import fid

REAL_DIR = "/mnt/project/data/real/all"
SIM_DIR = "/mnt/project/data/sim/real"


def main():
    score = fid.compute_fid(REAL_DIR, SIM_DIR, num_workers=0)
    print(f"(real) FID Score: {score}")

if __name__ == "__main__":
    main()