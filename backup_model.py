"""Snapshot the current trained model + logs + results + config before changing anything.
   python backup_model.py v1        -> backups/v1/ (with SHA-256 manifest)"""
import hashlib, os, shutil, sys, glob

label = sys.argv[1] if len(sys.argv) > 1 else "v1"
dst = os.path.join("backups", label)
if os.path.exists(dst):
    sys.exit(f"{dst} already exists - choose another label (never overwrite a backup).")
os.makedirs(dst)
files = (glob.glob("models/*.zip") + glob.glob("models/*.yaml") + glob.glob("logs/*.csv")
         + glob.glob("results/*.csv") + glob.glob("results/*.json") + glob.glob("results/*.png") + ["config.yaml"])
with open(os.path.join(dst, "MANIFEST.sha256"), "w") as m:
    for f in files:
        if os.path.isfile(f):
            out = os.path.join(dst, f)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(f, out)
            m.write(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {f}\n")
print(f"Backed up {len(files)} paths to {dst}/ (see MANIFEST.sha256)")
