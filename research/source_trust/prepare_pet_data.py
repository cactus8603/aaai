#!/usr/bin/env python3
"""Reproduce balanced Oxford-IIIT Pet pilot selections without model filtering."""
import argparse, collections, hashlib, json, random, sys, tarfile, urllib.request
import PIL
from pathlib import Path
from PIL import Image, ImageOps

SEED = 20261111
SOURCES = {'images.tar.gz':'5c4f3ee8e5d25df40f4fd59a7f44e54c','annotations.tar.gz':'95a8c909bbe2e81eed6a22bccdf3f68f'}
BASE = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/'
def digest(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',default='.cache/datasets/oxford-iiit-pet'); p.add_argument('--output',default='research/source_trust/natural_images_v1'); a=p.parse_args()
    cache=Path(a.cache); out=Path(a.output); cache.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    sources=[]
    for name,md5 in SOURCES.items():
        archive=cache/name
        if not archive.exists(): urllib.request.urlretrieve(BASE+name,archive)
        actual=digest(archive,'md5')
        if actual!=md5: raise RuntimeError(f'{archive}: MD5 {actual} != {md5}')
        sources.append({'url':BASE+name,'archive':str(archive),'bytes':archive.stat().st_size,'md5':actual,'sha256':digest(archive)})
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                dest=cache/member.name
                if member.issym() or member.islnk() or not dest.resolve().is_relative_to(cache.resolve()): raise RuntimeError(f'Unsafe tar member: {member.name}')
                if member.isdir(): dest.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    if not dest.exists():
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with tar.extractfile(member) as src, dest.open('wb') as dst:
                            import shutil
                            shutil.copyfileobj(src,dst)
                else: raise RuntimeError(f'Unsupported tar member: {member.name}')
    rng=random.Random(SEED); seen_bytes={}; seen_pixels={}; issues=[]; counts={}; rows_all=[]
    for split,official,n in [('dev128','trainval',64),('heldout256','test',128)]:
        breeds=collections.defaultdict(list)
        for line in (cache/'annotations'/f'{official}.txt').read_text().splitlines():
            if not line or line.startswith('#'):continue
            name,breed,species,species_breed=line.split()
            breeds[(int(species),int(breed))].append((name,int(species_breed)))
        rows=[]
        for species in [1,2]:
            keys=sorted(k for k in breeds if k[0]==species); rng.shuffle(keys)
            queues={k:sorted(breeds[k]) for k in keys}
            for q in queues.values(): rng.shuffle(q)
            needed={k:n//len(keys)+(i<n%len(keys)) for i,k in enumerate(keys)}
            for key in keys:
                accepted=0
                while accepted<needed[key]:
                    name,species_breed=queues[key].pop(); source=cache/'images'/f'{name}.jpg'
                    try:
                        with Image.open(source) as im:
                            im.load(); rgb=im.convert('RGB'); width,height=rgb.size
                            pixel_sha=hashlib.sha256(f'{width}x{height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
                    except Exception as e:
                        issues.append({'image_id':name,'reason':'decode_error','detail':str(e)});continue
                    sha=digest(source)
                    if sha in seen_bytes or pixel_sha in seen_pixels:
                        issues.append({'image_id':name,'reason':'exact_or_pixel_duplicate','duplicate_of':seen_bytes.get(sha,seen_pixels.get(pixel_sha))});continue
                    seen_bytes[sha]=name;seen_pixels[pixel_sha]=name
                    generated={}
                    for size in [224,448]:
                        dest=cache/f'letterbox{size}'/f'{name}.png';dest.parent.mkdir(exist_ok=True)
                        canvas=ImageOps.pad(rgb,(size,size),method=Image.Resampling.LANCZOS,color=(127,127,127),centering=(.5,.5));canvas.save(dest)
                        generated[str(size)]={'path':str(dest),'sha256':digest(dest)}
                    rows.append({'id':name,'split':split,'official_split':official,'label':'cat' if species==1 else 'dog','species_id':species,'breed_id':key[1],'species_breed_id':species_breed,'breed':name.rsplit('_',1)[0],'original_path':str(source),'original_sha256':sha,'rgb_pixel_sha256':pixel_sha,'original_width':width,'original_height':height,'images':generated})
                    accepted+=1
        rows.sort(key=lambda r:r['id']); manifest=out/f'{split}.jsonl';manifest.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));rows_all+=rows
        counts[split]={'total':len(rows),'classes':dict(collections.Counter(r['label'] for r in rows)),'breeds':dict(sorted(collections.Counter(r['breed'] for r in rows).items())),'manifest_sha256':digest(manifest)}
    dump(out/'readiness.json',{'ready':len(rows_all)==384,'dataset':'Oxford-IIIT Pet','seed':SEED,'runtime':{'python':sys.version.split()[0],'pillow':PIL.__version__},'script_sha256':digest(Path(__file__)),'official_annotation_sha256':{name:digest(cache/'annotations'/f'{name}.txt') for name in ['trainval','test']},'selection':'Python random.Random; sorted official annotations grouped by species/breed; shuffled breed order and candidates; equal breed quotas with seeded remainder; replace decode failures or duplicates only','license':'CC-BY-SA-4.0','license_source':'https://www.robots.ox.ac.uk/~vgg/data/pets/','license_verified_date':'2026-11-11','license_note':'Copyright remains with original image owners.','sources':sources,'splits':counts,'validation':{'selected_decode_pass':len(rows_all),'unique_original_sha256':len(seen_bytes),'unique_rgb_pixel_sha256':len(seen_pixels),'duplicates_checked':'Across and within all selected splits; RGB pixel hashes include dimensions; no model-based filtering','issues':issues},'preprocessing':{'sizes':[224,448],'method':'aspect preserving LANCZOS resize, centered gray RGB(127,127,127) letterbox via Pillow ImageOps.pad','orientation':'raw decoded orientation; no EXIF transpose'},'cache_bytes':sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())})
    print((out/'readiness.json').read_text())
if __name__=='__main__': main()
