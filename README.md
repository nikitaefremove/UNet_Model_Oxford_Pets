# Building UNet Model on OxfordIIITPet Dataset



```python
whole_train_valid_cycle(model, 15, 'UNET segmentation')
```

    Valid accuracy on 14 = 0.890987634333904



    
![png](README_files/README_22_1.png)
    



    
![png](README_files/README_22_2.png)
    



    
![png](README_files/README_22_3.png)
    



    
![png](README_files/README_22_4.png)
    



    
![png](README_files/README_22_5.png)
    



    
![png](README_files/README_22_6.png)
    



    
![png](README_files/README_22_7.png)
    



    
![png](README_files/README_22_8.png)
    



    
![png](README_files/README_22_9.png)
    



    
![png](README_files/README_22_10.png)
    



    
![png](README_files/README_22_11.png)
    



    
![png](README_files/README_22_12.png)
    


    Reached 89% accuracy on validation set. Stopping training.
    Valid accuracy = 0.890987634333904



```python
predictions = predict(model, valid_subset_loader, device).unsqueeze(1).to(torch.uint8)
```


```python
predictions.shape
```




    torch.Size([200, 1, 256, 256])




```python
torch.save(predictions, 'predictions.pt')
```

____
