document.addEventListener('DOMContentLoaded', () => {
    const cursor = document.getElementById('cursor');

    let cursorX = 0;
    let cursorY = 0;
    let targetX = 0;
    let targetY = 0;

    const speed = 0.1;

    document.addEventListener('mousemove', (event) => {
        targetX = event.pageX;
        targetY = event.pageY;
    });


    function animate() {
        cursorX += (targetX - cursorX) * speed;
        cursorY += (targetY - cursorY) * speed;

        cursor.style.left = `${cursorX + 10}px`;
        cursor.style.top = `${cursorY + 10}px`;

        requestAnimationFrame(animate);
    }

    animate();
});
