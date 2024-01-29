#! /usr/bin/env sh
# --recursive -r
# --times -t
# --dry-run -n
# Some flag explanations:
# --times: We must have this since rsync uses timestamps and filesize
#    to determine if the file needs to be synced, so if we don't
#    copy over modifications times it will copy everything every time.
# --delete-during: The most efficient way of doing deletions. We need
#    deletions because we want to mirror the files on the cluster, and
#    leaving old files in place could e.g. corrupt git repositories.
# --info=progress2: Better format for showing progress, doesn't spit a lot
#    of text onto the screen.
# --munge-links: Since this will be copying onto an NTFS filesystem the
#    symbolic links won't work anyway, but knowing they exist is still
#    useful.

rsync --info=progress2 --recursive --times --exclude='full' \
~/archive/2024/ ~/sync/data/slurm_jobs/
