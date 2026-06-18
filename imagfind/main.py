import os
import sys
import json
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import package modules
from .database import DatabaseManager
from .utils import scan_directory, format_size

# Consoles for standard output (results) and error stream (logs, progress, diagnostics)
console = Console()
err_console = Console(stderr=True)

def sync_index(db: DatabaseManager, recursive: bool, force: bool = False):
    """Scan the current working directory and incrementally update the ChromaDB index."""
    pwd = os.getcwd()
    
    # 1. Scan files on disk
    with err_console.status("[dim]Scanning directory for images...[/dim]", spinner="dots"):
        fs_images = scan_directory(pwd, recursive=recursive)
        
    # 2. Retrieve files in database
    db_data = db.get_all()
    db_ids = db_data["ids"] if db_data and "ids" in db_data else []
    db_metadatas = db_data["metadatas"] if db_data and "metadatas" in db_data else []
    
    db_map = {}
    for i in range(len(db_ids)):
        db_map[db_ids[i]] = db_metadatas[i]
        
    # 3. Detect changes
    to_delete = []
    for db_id in db_map:
        if db_id not in fs_images:
            # If recursive scan, it's deleted. 
            # If non-recursive, only delete if the file should have been in the top level (i.e. no '/' in relative path)
            if recursive or '/' not in db_id:
                to_delete.append(db_id)
                
    to_add = []
    to_update = []
    
    for rel_path, fs_info in fs_images.items():
        if rel_path not in db_map or force:
            to_add.append((rel_path, fs_info["abs_path"], fs_info))
        else:
            db_info = db_map[rel_path]
            fs_mtime = fs_info["mtime"]
            fs_size = fs_info["size_bytes"]
            
            db_mtime = float(db_info.get("mtime", 0))
            db_size = int(db_info.get("size_bytes", 0))
            
            if abs(fs_mtime - db_mtime) > 0.01 or fs_size != db_size:
                to_update.append((rel_path, fs_info["abs_path"], fs_info))
                
    # 4. Perform database updates
    if to_delete:
        db.delete_images(to_delete)
        err_console.print(f"[yellow]Sync: Removed {len(to_delete)} deleted images from index.[/yellow]")
        
    total_work = len(to_add) + len(to_update)
    if total_work > 0:
        # Load embedding model only when indexing work is required
        from .model import SiglipEmbedder
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
        
        with err_console.status("[bold cyan]Loading SigLIP model... (first run downloads ~200MB)"):
            embedder = SiglipEmbedder()
            
        err_console.print(f"[cyan]Syncing {total_work} image files ({len(to_add)} new, {len(to_update)} modified)...")
        
        ids = []
        embeddings = []
        metadatas = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err_console
        ) as progress:
            task = progress.add_task("[cyan]Embedding images", total=total_work)
            
            all_items = to_add + to_update
            for rel_path, abs_path, fs_info in all_items:
                try:
                    from PIL import Image
                    with Image.open(abs_path) as img:
                        pil_img = img.convert("RGB")
                        width, height = pil_img.size
                        embedding = embedder.get_image_embedding(pil_img)
                        
                    metadata = {
                        "filename": fs_info["filename"],
                        "abs_path": abs_path,
                        "width": width,
                        "height": height,
                        "size_bytes": fs_info["size_bytes"],
                        "mtime": fs_info["mtime"]
                    }
                    
                    ids.append(rel_path)
                    embeddings.append(embedding)
                    metadatas.append(metadata)
                    
                    # Batch inserts to improve database performance
                    if len(ids) >= 50:
                        db.add_images(ids, embeddings, metadatas)
                        ids, embeddings, metadatas = [], [], []
                        
                except Exception as e:
                    err_console.print(f"[red]✕ Failed to index {rel_path}: {str(e)}[/red]")
                    
                progress.update(task, advance=1)
                
            # Insert any remaining records
            if ids:
                db.add_images(ids, embeddings, metadatas)
                
        err_console.print("[green]✓ Index synchronization complete![/green]")
    else:
        # If there were deletions but no additions, we already printed a message.
        # Otherwise, just print a quiet status message.
        if not to_delete:
            err_console.print("[dim]Index is up to date.[/dim]")

def open_file_in_viewer(filepath: str):
    """Opens a file using the system default viewer."""
    import platform
    import subprocess
    
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        err_console.print(f"[red]✕ File not found: {filepath}[/red]")
        return
        
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(abs_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", abs_path], check=True)
        else:  # Linux / other Unix
            subprocess.run(["xdg-open", abs_path], check=True)
        err_console.print(f"[green]✓ Opening '{filepath}' in default viewer...[/green]")
    except Exception as e:
        err_console.print(f"[red]✕ Failed to open file: {str(e)}[/red]")

def generate_and_open_preview(results: list):
    """Generates a temporary grid image of top search results showing ranks and similarity scores, and opens it."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import tempfile
    
    thumbnails = []
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    def draw_banner_text(draw_obj, text, banner_y, banner_height, color=(255, 255, 255)):
        if font:
            try:
                bbox = draw_obj.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                try:
                    text_w, text_h = draw_obj.textsize(text, font=font)
                except Exception:
                    text_w, text_h = len(text) * 6, 10
        else:
            text_w, text_h = len(text) * 6, 10
            
        x = (300 - text_w) // 2
        y = banner_y + (banner_height - text_h) // 2
        draw_obj.text((x, y), text, fill=color, font=font)

    for rank, res in enumerate(results, 1):
        img_path = res["metadata"].get("abs_path")
        if not img_path or not os.path.exists(img_path):
            img_path = os.path.abspath(res["id"])
            
        thumb = None
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    thumb = ImageOps.pad(img.convert("RGB"), (300, 300), color=(40, 40, 40))
            except Exception:
                pass
                
        if thumb is None:
            thumb = Image.new("RGB", (300, 300), color=(80, 80, 80))
            draw = ImageDraw.Draw(thumb)
            if font:
                try:
                    bbox = draw.textbbox((0, 0), "Missing Image", font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    try:
                        text_w, text_h = draw.textsize("Missing Image", font=font)
                    except Exception:
                        text_w, text_h = 70, 10
            else:
                text_w, text_h = 70, 10
            x = (300 - text_w) // 2
            y = (300 - text_h) // 2
            draw.text((x, y), "Missing Image", fill=(200, 200, 200), font=font)
            
        overlay = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, 260, 300, 300], fill=(0, 0, 0, 180))
        text_str = f"#{rank}  {res['similarity'] * 100:.1f}%"
        draw_banner_text(overlay_draw, text_str, 260, 40)
        
        thumb_rgba = thumb.convert("RGBA")
        thumb = Image.alpha_composite(thumb_rgba, overlay).convert("RGB")
        thumbnails.append(thumb)
        
    num_results = len(results)
    cols = min(3, num_results)
    rows = (num_results + cols - 1) // cols
    
    grid_w = cols * 300
    grid_h = rows * 300
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    
    for idx, thumb in enumerate(thumbnails):
        r = idx // cols
        c = idx % cols
        grid_img.paste(thumb, (c * 300, r * 300))
        
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = tmp_file.name
            grid_img.save(temp_path)
        err_console.print(f"[green]✓ Saved grid preview to temporary file: {temp_path}[/green]")
        open_file_in_viewer(temp_path)
    except Exception as e:
        err_console.print(f"[red]✕ Failed to generate preview grid: {str(e)}[/red]")

@click.command()
@click.option('-t', '--text', type=str, help="Search the index with a text query.")
@click.option('-i', '--image', type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Search the index with a query image path.")
@click.option('--index', is_flag=True, help="Force scan and update index in the current directory.")
@click.option('--recursive', '-r', is_flag=True, help="Scan directories recursively.")
@click.option('--list-images', '--list', is_flag=True, help="List all currently indexed images.")
@click.option('--delete', type=str, help="Delete an image from the index by relative path/ID.")
@click.option('--clear', is_flag=True, help="Reset/Clear the entire index for this directory.")
@click.option('--info', is_flag=True, help="Show database location and stats.")
@click.option('--limit', '-n', type=int, default=5, help="Max number of search results to return.")
@click.option('--threshold', type=float, default=0.0, help="Minimum similarity threshold (0.0 to 1.0) to filter results.")
@click.option('--json', 'output_json', is_flag=True, help="Format search results as raw JSON to stdout.")
@click.option('--force', is_flag=True, help="Force rebuild embeddings even for unmodified files.")
@click.option('--open', 'open_result', is_flag=True, help="Automatically open the top search result in the default image viewer.")
@click.option('--preview', is_flag=True, help="Generate and open a grid preview of top search results.")
@click.option('--duplicates', is_flag=True, help="Identify duplicate or near-duplicate images in the folder.")
@click.option('--csv', 'csv_path', type=click.Path(writable=True, file_okay=True, dir_okay=False), help="Export search results to a CSV file.")
@click.option('--stats', is_flag=True, help="Show folder-wise statistics for all indexed images.")
def main(text, image, index, recursive, list_images, delete, clear, info, limit, threshold, output_json, force, open_result, preview, duplicates, csv_path, stats):
    """AI-powered semantic image search CLI using SigLIP and ChromaDB."""
    
    # 1. Check if no arguments are passed
    if not any([text, image, index, list_images, delete, clear, info, duplicates, stats]):
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()
        
    # 2. Connect to ChromaDB in the PWD
    db = DatabaseManager(".imagfind_db")
    
    # 3. Execute requested actions
    if clear:
        if click.confirm("[yellow]Are you sure you want to clear the entire search index for this directory?[/yellow]", default=False):
            db.clear_all()
            err_console.print("[green]✓ Index cleared successfully![/green]")
        return
        
    if delete:
        db.delete_images([delete])
        err_console.print(f"[green]✓ Deleted '{delete}' from index.[/green]")
        return
        
    # All other operations require directory sync first
    sync_index(db, recursive=recursive, force=(force or index))
    
    if index:
        # We already synced above, so just exit
        return
        
    if info:
        stats = db.get_stats()
        # Find execution device
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        
        info_text = (
            f"[bold]Model:[/bold] google/siglip-base-patch16-224 ({device})\n"
            f"[bold]Indexed Items:[/bold] {stats['count']} images\n"
            f"[bold]DB Location:[/bold] {stats['db_path']}"
        )
        err_console.print(Panel(info_text, title="imagfind: CLI Image Search Diagnostics", border_style="cyan", expand=False))
        return
        
    if list_images:
        data = db.get_all()
        ids = data["ids"] if data and "ids" in data else []
        metadatas = data["metadatas"] if data and "metadatas" in data else []
        
        if not ids:
            err_console.print("[yellow]No images currently indexed. Perform a search or scan directory to populate index.[/yellow]")
            return
            
        table = Table(title=f"Indexed Images (Total: {len(ids)})", border_style="dim")
        table.add_column("Relative Path / ID", style="cyan")
        table.add_column("Dimensions", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Last Modified", style="dim")
        
        from datetime import datetime
        for i in range(len(ids)):
            meta = metadatas[i]
            w, h = meta.get("width", 0), meta.get("height", 0)
            dim_str = f"{w}x{h}" if w and h else "unknown"
            size_str = format_size(int(meta.get("size_bytes", 0)))
            mtime_val = float(meta.get("mtime", 0))
            mtime_str = datetime.fromtimestamp(mtime_val).strftime("%Y-%m-%d %H:%M:%S") if mtime_val else "unknown"
            
            table.add_row(ids[i], dim_str, size_str, mtime_str)
        console.print(table)
        return
        
    if stats:
        stats_info = db.get_stats()
        count = stats_info["count"]
        db_path = stats_info["db_path"]
        from .utils import get_dir_size
        db_size_bytes = get_dir_size(db_path)
        db_size_str = format_size(db_size_bytes)
        
        if count == 0:
            table = Table(title="imagfind: Folder Statistics", border_style="cyan")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value")
            table.add_row("Total Indexed Images", "0")
            table.add_row("Database Size on Disk", db_size_str)
            console.print(table)
            return
            
        data = db.get_all()
        ids = data["ids"] if data and "ids" in data else []
        metadatas = data["metadatas"] if data and "metadatas" in data else []
        
        total_size_bytes = 0
        total_width = 0
        total_height = 0
        valid_res_count = 0
        
        largest_size = -1
        largest_path = "N/A"
        smallest_size = float('inf')
        smallest_path = "N/A"
        
        from collections import Counter
        extensions = []
        
        for rel_path, meta in zip(ids, metadatas):
            size = int(meta.get("size_bytes", 0))
            total_size_bytes += size
            
            w = int(meta.get("width", 0))
            h = int(meta.get("height", 0))
            if w > 0 and h > 0:
                total_width += w
                total_height += h
                valid_res_count += 1
                
            if size > largest_size:
                largest_size = size
                largest_path = rel_path
            if size < smallest_size:
                smallest_size = size
                smallest_path = rel_path
                
            _, ext = os.path.splitext(rel_path)
            if ext:
                extensions.append(ext.lower())
                
        avg_res_str = "N/A"
        if valid_res_count > 0:
            avg_w = round(total_width / valid_res_count)
            avg_h = round(total_height / valid_res_count)
            avg_res_str = f"{avg_w}x{avg_h}"
            
        largest_str = "N/A"
        if largest_size >= 0:
            largest_str = f"{largest_path} ({format_size(largest_size)})"
            
        smallest_str = "N/A"
        if smallest_size != float('inf'):
            smallest_str = f"{smallest_path} ({format_size(smallest_size)})"
            
        most_common_ext = "N/A"
        if extensions:
            most_common_ext, ext_count = Counter(extensions).most_common(1)[0]
            most_common_ext = f"{most_common_ext} ({ext_count} files)"
            
        table = Table(title="imagfind: Folder Statistics", border_style="cyan")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value")
        
        table.add_row("Total Indexed Images", str(count))
        table.add_row("Total Image Storage Size", format_size(total_size_bytes))
        table.add_row("Average Image Resolution", avg_res_str)
        table.add_row("Largest Image File", largest_str)
        table.add_row("Smallest Image File", smallest_str)
        table.add_row("Database Size on Disk", db_size_str)
        table.add_row("Most Common Extension", most_common_ext)
        
        console.print(table)
        return

    if duplicates:
        data = db.get_all_with_embeddings()
        ids = data["ids"] if data and "ids" in data else []
        metadatas = data["metadatas"] if data and "metadatas" in data else []
        embeddings = data["embeddings"] if data and "embeddings" in data else []
        
        if not ids:
            err_console.print("[yellow]No images currently indexed. Sync or index a directory first.[/yellow]")
            return
            
        dup_threshold = threshold if threshold > 0.0 else 0.90
        
        import numpy as np
        embeds = np.array(embeddings)
        sim_matrix = np.dot(embeds, embeds.T)
        
        visited = set()
        groups = []
        for i in range(len(ids)):
            if i not in visited:
                group = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    group.append(curr)
                    for neighbor in range(len(ids)):
                        if neighbor not in visited and sim_matrix[curr, neighbor] >= dup_threshold:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(group) > 1:
                    group.sort(key=lambda idx: ids[idx])
                    groups.append(group)
                    
        if not groups:
            console.print(f"[yellow]No duplicate image groups found (threshold: {dup_threshold:.2f}).[/yellow]")
            return
            
        table = Table(title=f"Duplicate Image Groups (Threshold: {dup_threshold:.2f})")
        table.add_column("Group", justify="center", style="bold cyan")
        table.add_column("File Name", style="bold")
        table.add_column("Relative Path", style="dim")
        table.add_column("Size", justify="right")
        table.add_column("Similarity", justify="right")
        
        for g_idx, group in enumerate(groups, 1):
            rep_idx = group[0]
            rep_id = ids[rep_idx]
            rep_meta = metadatas[rep_idx]
            rep_size = format_size(int(rep_meta.get("size_bytes", 0)))
            
            table.add_row(
                f"Group {g_idx}",
                rep_meta.get("filename", os.path.basename(rep_id)),
                rep_id,
                rep_size,
                "Representative"
            )
            
            for member_idx in group[1:]:
                m_id = ids[member_idx]
                m_meta = metadatas[member_idx]
                m_size = format_size(int(m_meta.get("size_bytes", 0)))
                sim_val = sim_matrix[rep_idx, member_idx]
                
                table.add_row(
                    "",
                    m_meta.get("filename", os.path.basename(m_id)),
                    m_id,
                    m_size,
                    f"{sim_val * 100:.1f}%"
                )
                
            if g_idx < len(groups):
                table.add_row("", "", "", "", "")
                
        console.print(table)
        return

    # Perform searches
    if text or image:
        from .model import SiglipEmbedder
        from PIL import Image
        
        results = []
        if text:
            with err_console.status("[bold cyan]Loading SigLIP text model..."):
                embedder = SiglipEmbedder()
            query_embedding = embedder.get_text_embedding(text)
            results = db.search(query_embedding, limit=limit)
        elif image:
            try:
                with Image.open(image) as img:
                    pil_img = img.convert("RGB")
                    with err_console.status("[bold cyan]Loading SigLIP vision model..."):
                        embedder = SiglipEmbedder()
                    query_embedding = embedder.get_image_embedding(pil_img)
                results = db.search(query_embedding, limit=limit)
            except Exception as e:
                err_console.print(f"[red]✕ Error opening query image {image}: {str(e)}[/red]")
                sys.exit(1)
                
        # Scale and format similarity scores
        for res in results:
            raw_sim = res["similarity"]
            if text:
                # Map cross-modal cosine similarity [-0.1, 0.2] to [0.0, 1.0]
                scaled_sim = max(0.0, min(1.0, (raw_sim + 0.1) / 0.3))
            else:
                # Keep raw image-to-image similarity bounded between 0 and 1
                scaled_sim = max(0.0, min(1.0, raw_sim))
            
            res["similarity"] = scaled_sim
            res["raw_similarity"] = raw_sim
                
        # Filter results by threshold
        if threshold > 0.0:
            results = [r for r in results if r["similarity"] >= threshold]
            
        # Export CSV if requested
        if csv_path:
            import csv
            try:
                with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Rank", "Similarity", "File Path", "Width", "Height", "File Size"])
                    for rank, res in enumerate(results, 1):
                        meta = res["metadata"]
                        writer.writerow([
                            rank,
                            f"{res['similarity'] * 100:.1f}%",
                            res["id"],
                            meta.get("width", "unknown"),
                            meta.get("height", "unknown"),
                            meta.get("size_bytes", "unknown")
                        ])
                err_console.print(f"[green]✓ Exported {len(results)} results to CSV: {csv_path}[/green]")
            except Exception as e:
                err_console.print(f"[red]✕ Failed to export CSV: {str(e)}[/red]")

        # Automatically open top search result if --open is specified
        if open_result:
            if results:
                top_res = results[0]
                abs_path = top_res["metadata"].get("abs_path")
                if not abs_path or not os.path.exists(abs_path):
                    abs_path = os.path.abspath(top_res["id"])
                open_file_in_viewer(abs_path)
            else:
                err_console.print("[yellow]No results to open.[/yellow]")

        # Generate and open preview grid if --preview is specified
        if preview:
            if results:
                generate_and_open_preview(results)
            else:
                err_console.print("[yellow]No results to preview.[/yellow]")
                
        if output_json:
            # Output only raw JSON to stdout (suppresses visual tables)
            print(json.dumps(results, indent=2))
        else:
            if not results:
                console.print("[yellow]No matches found.[/yellow]")
                return
                
            # Table formatting
            table = Table(title=f"Search Results (Query: '{text or image}')")
            table.add_column("Rank", justify="center")
            table.add_column("Similarity", justify="right")
            table.add_column("File Path", style="cyan")
            table.add_column("Dimensions", justify="right")
            table.add_column("Size", justify="right")
            
            for rank, res in enumerate(results, 1):
                sim = res["similarity"]
                sim_percent = sim * 100
                if sim_percent >= 75:
                    sim_str = f"[bold green]{sim_percent:.1f}%[/bold green]"
                elif sim_percent >= 50:
                    sim_str = f"[bold yellow]{sim_percent:.1f}%[/bold yellow]"
                else:
                    sim_str = f"[bold red]{sim_percent:.1f}%[/bold red]"
                    
                meta = res["metadata"]
                w, h = meta.get("width", 0), meta.get("height", 0)
                dim_str = f"{w}x{h}" if w and h else "unknown"
                size_str = format_size(int(meta.get("size_bytes", 0)))
                
                table.add_row(str(rank), sim_str, res["id"], dim_str, size_str)
                
            console.print(table)

if __name__ == "__main__":
    main()
